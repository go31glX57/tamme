import polars as pl
import torch
from omegaconf import DictConfig
from torch import nn

from model.projector import MultiModalInputProjector, MultiModalOutputProjector
from model.transformer import TransformerEncoder, TransformerDecoder
from utils import sample


class MAE(nn.Module):

    def __init__(self, cfg: DictConfig):
        """
        A Transformer Encoder and Decoder in a Masked Autoencoder Setup.
        :param cfg: the global configuration.
        """
        super(MAE, self).__init__()
        self.cfg = cfg

        self.type_cat_freq = nn.Parameter(torch.tensor(pl.read_csv(cfg.type_cat_freq)['count']), requires_grad=False)

        self.input_projector = MultiModalInputProjector(cfg, return_inputs=True)

        self.encoder = TransformerEncoder(cfg)

        self.encoder_to_decoder = nn.Linear(cfg.emb_dim, cfg.emb_dim_dec)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, cfg.emb_dim_dec))

        self.decoder = TransformerDecoder(cfg)

        self.output_projector = MultiModalOutputProjector(cfg)

        nn.init.normal_(self.mask_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        # x: (B, S, E)
        # pos_idx: (B, S, 1)
        # attn_mask: (B, 1, 1, S)
        # t: (B, S, F_T)
        # modality: (B, S)
        x, pos_idx, attn_mask, t, inputs = self.input_projector(x)

        B, S, _ = x.shape

        n_mask = max(int(S * self.cfg.mask_ratio), 1)

        # Biased random masking based on high-level type category.
        _types_cat = inputs['types_cat']
        assert _types_cat.size(1) == S
        # (n_types_cat + 1), insert prob = 0 for padding -1
        _types_cat_probs = torch.cat([1 / self.type_cat_freq, torch.zeros(1, device=self.type_cat_freq.device)])
        # (B, n_mask), (B, S - n_mask)
        masked_idx, sel_idx = sample(_types_cat, n_mask, _types_cat_probs)
        # (B, n_mask, 1), (B, S - n_mask, 1)
        masked_idx = masked_idx.unsqueeze(-1)
        sel_idx = sel_idx.unsqueeze(-1)

        # (B, S)
        unshuffle_indices = torch.cat([masked_idx, sel_idx], dim=1).argsort(dim=1).squeeze(-1)

        # (B, n_mask, E)
        x_masked = x.gather(1, masked_idx.expand(-1, -1, x.size(-1)))
        # (B, S - n_mask, E)
        x_sel = x.gather(1, sel_idx.expand(-1, -1, x.size(-1)))

        # (B, n_mask, 1)
        pos_idx_masked = pos_idx.gather(1, masked_idx)
        # (B, S - n_mask, 1)
        pos_idx_sel = pos_idx.gather(1, sel_idx)

        # (B, 1, 1, n_mask)
        attn_mask_masked = attn_mask.gather(-1, masked_idx.permute(0, 2, 1).unsqueeze(1))
        # (B, 1, 1, S - n_mask)
        attn_mask_sel = attn_mask.gather(-1, sel_idx.permute(0, 2, 1).unsqueeze(1))

        # (B, n_mask, F_T)
        t_masked = t.gather(1, masked_idx.expand(-1, -1, t.size(-1)))
        # (B, S - n_mask, F_T)
        t_sel = t.gather(1, sel_idx.expand(-1, -1, t.size(-1)))

        # (B, S - n_mask + 1, E) with cls token
        latent = self.encoder(x_sel, pos_idx_sel, attn_mask_sel, t_sel)

        # (B, S - n_mask + 1, E') with cls token
        latent = self.encoder_to_decoder(latent)
        # (B, 1, E')
        cls_tokens = latent[:, 0].unsqueeze(1)
        # (B, S - n_mask, E')
        latent = latent[:, 1:, :]

        # (B, n_mask, E')
        mask_tokens = self.mask_token.repeat(x.size(0), x_masked.size(1), 1)

        # (B, S, E')
        latent = torch.cat([latent, mask_tokens], dim=1)
        # (B, S, 1)
        pos_idx_all = torch.cat([pos_idx_sel, pos_idx_masked], 1)
        # (B, S, F_T)
        t_all = torch.cat([t_sel, t_masked], 1)
        # (B, 1, 1, S)
        attn_mask_all = torch.cat([attn_mask_sel, attn_mask_masked], -1)

        # (B, S + 1, E') with cls token
        x_hat = self.decoder(latent, pos_idx_all, attn_mask_all, t_all, cls_tokens)
        # (B, S, E')
        x_hat = x_hat[:, 1:, :]
        # (B, S), 1 = masked, 0 = unmasked
        mask = torch.cat([torch.ones_like(masked_idx), torch.zeros_like(sel_idx)], 1).squeeze(-1)

        x_hat_unshuffled = x_hat.gather(1, unshuffle_indices.unsqueeze(-1).expand(-1, -1, x_hat.size(-1)))
        mask_unshuffled = mask.gather(1, unshuffle_indices)

        recon = self.output_projector(x_hat_unshuffled, inputs['splits'])

        padding_mask = attn_mask.squeeze(1, 2)

        return inputs, recon, padding_mask, mask_unshuffled
