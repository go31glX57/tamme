from functools import partial

import torch
from torch import nn

from model.transformer.building_blocks import Block
from model.transformer.pos_embedding import PosEmbedding
from utils import local_window_shuffle


class TransformerEncoder(nn.Module):
    def __init__(self, cfg):
        super(TransformerEncoder, self).__init__()

        self.cfg = cfg

        self.cls_token = nn.Parameter(torch.randn(1, 1, cfg.emb_dim))
        self.pos_emb = PosEmbedding(cfg, cfg.emb_dim)

        self.pos_jitter = partial(local_window_shuffle, w=cfg.pos_jitter_window_size, p=cfg.pos_jitter_prob)

        self.emb_dropout = nn.Dropout(cfg.emb_dropout)

        # stochastic depth decay
        dpr = [x.item() for x in torch.linspace(0, cfg.drop_path_rate, cfg.n_layers)]

        self.blocks = nn.ModuleList([
            Block(
                dim=cfg.emb_dim,
                num_heads=cfg.n_heads,
                mlp_ratio=cfg.mlp_ratio,
                qkv_bias=True,
                drop=cfg.dropout,
                attn_drop=cfg.attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                act_layer=nn.GELU
            )
            for i in range(cfg.n_layers)
        ])

        torch.nn.init.normal_(self.cls_token, std=.02)

    def forward(self, x, pos_idx, attn_mask, time):
        # Build pos emb.
        pe = self.pos_emb(pos_idx.squeeze(-1), time)

        # Apply pos jitter.
        if self.training:
            pe_idx = torch.arange(pe.size(1), device=pe.device)
            pe_idx = self.pos_jitter(pe_idx)
            pe = pe.index_select(1, pe_idx)

        # Add pos emb.
        x = x + pe

        # Apply dropout.
        x = self.emb_dropout(x)

        # Add cls token.
        x = torch.cat([self.cls_token.repeat(x.size(0), 1, 1), x], 1)
        cls_mask = torch.ones(attn_mask.size(0), 1, 1, 1, device=attn_mask.device, dtype=attn_mask.dtype)
        attn_mask = torch.cat([cls_mask, attn_mask], dim=-1)

        # Actual forward pass.
        for i, blk in enumerate(self.blocks):
            x = blk(x, attn_mask=attn_mask)

        return x


    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

    def freeze_first_layers(self, n):
        """
        n == -1 -> all layers frozen except last
        """
        if n < -1: return None
        if n == -1: n = len(self.blocks) - 1

        for i, block in enumerate(self.blocks):
            if i < n:
                for param in block.parameters():
                    param.requires_grad = False
                block.eval()
            else:
                for param in block.parameters():
                    param.requires_grad = True
                block.train()


    def unfreeze_layer(self, i):
        block = self.blocks[i]
        for param in block.parameters():
            param.requires_grad = True
        block.train()