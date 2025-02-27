import math

import torch
from torch import nn

from model.projector import MultiModalInputProjector
from model.transformer import TransformerEncoder
from utils import replace_col


class Classifier(nn.Module):
    def __init__(self, cfg):
        super(Classifier, self).__init__()
        self.cfg = cfg

        self.input_projector = MultiModalInputProjector(cfg)

        self.summary_token = nn.Parameter(torch.randn(1, 1, cfg.emb_dim))

        self.encoder = TransformerEncoder(cfg)

        output_dim = cfg.n_multilabel if cfg.n_multilabel else (cfg.n_classes if cfg.n_classes > 2 else 1)
        self.head = nn.Linear(cfg.emb_dim, output_dim)

        self.apply(self._init_weights)
        torch.nn.init.normal_(self.summary_token, std=0.02)

        self._setup_finetuning()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _setup_finetuning(self) -> None:
        match getattr(self.cfg, 'finetune_mode', None):
            case None, 'full':
                pass
            case 'head_only':
                self.encoder.freeze()
                self.input_projector.freeze()
            case 'encoder_only':
                self.input_projector.freeze()
                self.freeze_head()
            case 'layers_after_n':
                self.encoder.freeze_first_layers(self.cfg.freeze_encoder_layers)
            case _:
                raise ValueError("Unknown finetuning mode")

        if getattr(self.cfg, 'freeze_input_projector', False):
            self.input_projector.freeze()
        if getattr(self.cfg, 'freeze_embeddings', False):
            self.input_projector.freeze_type_cat_emb()

    def freeze_head(self):
        for param in self.head.parameters():
            param.requires_grad = False

    def get_trainable_parameters(self):
        trainable_params = []
        frozen_params = []

        for param in self.parameters():
            if param.requires_grad:
                trainable_params.append(param)
            else:
                frozen_params.append(param)

        return trainable_params, frozen_params

    def forward_features(self, x, pool='cls'):
        x, pos_idx, attn_mask, time = self.input_projector(x)

        # Sort by pos idx, trim padding.
        # This will destroy modality order, but it is not needed here anyway since classifier does no reconstruction.
        pos_idx, sort_idx = pos_idx.sort(1)
        trim_idx = (pos_idx >= 0).int().argmax(1).min()

        pos_idx = pos_idx[:, trim_idx:].squeeze(-1)
        x = x.gather(1, sort_idx.repeat(1, 1, x.size(-1)))[:, trim_idx:]
        time = time.gather(1, sort_idx.repeat(1, 1, time.size(-1)))[:, trim_idx:]
        attn_mask = attn_mask.gather(-1, sort_idx.permute(0, 2, 1).unsqueeze(1))[..., trim_idx:]

        # Apply max length.
        m = self.cfg.max_effective_seq_length
        x, pos_idx, attn_mask, time = x[:, :m, ...], pos_idx[:, :m], attn_mask[..., :m], time[:, :m, ...]

        # Summarize long sequences.
        x, pos_idx, attn_mask, time = self._summarize(x, pos_idx, attn_mask, time)

        # Forward pass.
        x = self.encoder(x, pos_idx, attn_mask, time)

        # Extract Features.
        match pool:
            case 'cls':
                return x[:, 0]
            case 'mean':
                return x[:, 1:].mean(1, keepdim=True)
            case _:
                raise ValueError()

    def forward(self, x):
        x = self.forward_features(x, pool='cls')

        logits = self.head(x)

        return logits

    def _summarize(self, x, pos_idx, attn_mask, time):
        s_max = self.cfg.seq_length
        B, S, E = x.shape
        overhead = S - s_max

        if overhead > 0:
            n_summaries = math.ceil(overhead / (s_max - 1))
            n_keep = s_max - n_summaries

            # Calculate splits. Most recent data to preserve is last.
            splits = [s_max] * (n_summaries - 1) + [n_keep]
            splits = [S - sum(splits)] + splits

            # Split data, update pos idx.
            chunks_x = x.split(splits, 1)
            chunks_pos = [
                (p - p.float().masked_fill(p < 0, float('inf')).min(-1)[0].unsqueeze(-1).broadcast_to(p.shape))
                .int().clamp(-1)
                for p in pos_idx.split(splits, 1)
            ]
            chunks_attn = attn_mask.split(splits, -1)
            chunks_t = time.split(splits, 1)

            # Encode chunks.
            summaries = []
            for i in range(n_summaries):
                out = self.encoder(chunks_x[i], chunks_pos[i], chunks_attn[i], chunks_t[i])

                # take cls token
                f = out[:, 0].unsqueeze(1)
                f = f + self.summary_token

                summaries.append(f)

            summaries = torch.cat(summaries, 1)

            # Build pos idx, attn mask, etc. for summaries.
            pos_sum = torch.arange(0, n_summaries, dtype=pos_idx.dtype, device=pos_idx.device).unsqueeze(0).repeat(B, 1)
            attn_sum = torch.stack([c.max(-1)[0] for c in chunks_attn[:-1]], -1)
            # caution! t_sum relies on the order of time features: "age", "pos_idx", "time_delta_ref", "time_delta_prev"
            t_sum = torch.stack([c[:,-1] for c in chunks_t[:-1]], 1)
            t_sum = replace_col(t_sum, 1, pos_sum)
            t_sum = replace_col(t_sum, 3, torch.stack([c[...,:, 3].clamp(0).sum(-1) * min(i, 1) for i, c in enumerate(chunks_t[:-1])], 1) )

            # Add pos idx, attn_mask, etc. for summaries.
            x_new = torch.cat([summaries, chunks_x[-1]], 1)
            pos_keep = torch.where(chunks_pos[-1] > -1, chunks_pos[-1] + n_summaries, -1)
            pos_new = torch.cat([pos_sum, pos_keep], 1)
            attn_new = torch.cat([attn_sum, chunks_attn[-1]], -1)
            t_keep_new = replace_col(chunks_t[-1], 1, pos_keep)

            t_new = torch.cat([t_sum, t_keep_new], 1)

            return x_new, pos_new, attn_new, t_new

        return x, pos_idx, attn_mask, time
