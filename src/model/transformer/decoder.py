from functools import partial

import torch
from torch import nn

from model.transformer.building_blocks import Block
from model.transformer.pos_embedding import PosEmbedding


class TransformerDecoder(nn.Module):
    def __init__(self, cfg):
        super(TransformerDecoder, self).__init__()
        self.cfg = cfg

        self.pos_emb = PosEmbedding(cfg, cfg.emb_dim_dec)

        self.blocks = nn.ModuleList(cfg.n_layers_dec * [
            Block(
                dim=cfg.emb_dim_dec,
                num_heads=cfg.n_heads_dec,
                mlp_ratio=cfg.mlp_ratio,
                qkv_bias=True,
                drop=cfg.dropout,
                attn_drop=cfg.attn_drop_rate,
                drop_path=0,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                act_layer=nn.GELU
            )
        ])

    def forward(self, x, pos_idx, attn_mask, time, cls_tokens):
        # x:  (B, S, E')
        # pos_idx: (B, S, 1)
        # attn_mask: (B, 1, 1, S)
        # t: (B, S, F_T)
        # cls_tokens: (B, 1, E')

        x = x + self.pos_emb(pos_idx.squeeze(-1), time)

        # (B, S + 1, E') with cls token
        x = torch.cat([cls_tokens, x], 1)

        # (B, 1, 1, 1)
        cls_mask = torch.ones(attn_mask.size(0), 1, 1, 1, device=attn_mask.device, dtype=attn_mask.dtype)
        # (B, 1, 1, S + 1)
        attn_mask = torch.cat([cls_mask, attn_mask], dim=-1)

        for i, blk in enumerate(self.blocks):
            x = blk(x, attn_mask=attn_mask)

        return x
