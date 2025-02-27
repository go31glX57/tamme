import math

import torch
from einops import rearrange
from torch import nn


def _make_sincos_position_embedding(seq_len, emb_dim):
    position = torch.arange(seq_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, emb_dim, 2).float() * (-math.log(10000.0) / emb_dim))

    pe = torch.zeros(seq_len, emb_dim)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    pe = pe.unsqueeze(0)

    pe = nn.Parameter(pe)
    pe.requires_grad = False

    return pe


class PosFeatureEmbedding(nn.Module):
    def __init__(self, n_features, emb_dim, temperature=10000):
        super(PosFeatureEmbedding, self).__init__()
        self.norm = nn.LayerNorm(n_features, eps=1e-6)

        assert emb_dim % (n_features * 2) == 0, "Embedding dimension must be divisible by 2 * n_features"
        pos_dim = emb_dim // (n_features * 2)

        omega = 1 / (temperature ** (torch.arange(pos_dim, dtype=torch.float32) / pos_dim))
        self.register_buffer('omega', omega)

    def forward(self, x):
        """
        :param x: Shape (B, S, n_features)
        """
        x = self.norm(x)
        x = torch.einsum('bsf,p->bsfp', [x, self.omega])
        x_sin = torch.sin(x)
        x_cos = torch.cos(x)
        pos_emb = torch.cat((x_sin, x_cos), dim=-1)  # B, s, f, 2p
        pos_emb = rearrange(pos_emb, 'b s f p -> b s (f p)')
        return pos_emb


class PosEmbedding(nn.Module):
    def __init__(self, cfg, emb_dim):
        super(PosEmbedding, self).__init__()
        self.cfg = cfg

        match cfg.pos_emb:
            case 'learnable':
                self.pos_emb = nn.Parameter(torch.randn(1, cfg.seq_length, emb_dim))
            case 'positional':
                self.pos_emb = _make_sincos_position_embedding(cfg.seq_length, emb_dim, )
            case 'temporal':
                self.pos_emb = PosFeatureEmbedding(cfg.n_time_features, emb_dim)
            case 'none' | None:
                self.pos_emb = nn.Parameter(torch.zeros((1, cfg.seq_length, emb_dim), dtype=torch.float32),
                                            requires_grad=False)
            case _:
                raise NotImplementedError("Unknown pos embedding technique.")

    def forward(self, pos_idx, time):
        if self.cfg.pos_emb == 'temporal':
            return self.pos_emb(time)
        else:
            return self.pos_emb[0, pos_idx.int()]
