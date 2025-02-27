from torch import nn
from timm.layers import DropPath, Mlp

NEGINF = float('-inf')


class Attention(nn.Module):

    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self._attn = None
        self._attn_gradients = None

    def forward(self, x, mask=None):
        """
        :param x: Shape (B, N, C)
        :param mask: 0 = ignored, 1 = attention enabled
        """

        self._attn = None
        self._attn_gradients = None

        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0], qkv[1], qkv[2]

        attn_scores = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn_scores = attn_scores.masked_fill(~mask, NEGINF)

        attn_probs = attn_scores.softmax(dim=-1)
        self._attn = attn_probs

        if attn_probs.requires_grad:
            attn_probs.register_hook(self._save_attn_grad)

        attn_probs = self.attn_drop(attn_probs)

        x = (attn_probs @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

    def _save_attn_grad(self, grad):
        self._attn_gradients = grad

    def get_attn(self):
        return self._attn

    def get_attn_grad(self):
        return self._attn_gradients


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()

        self.norm1 = norm_layer(dim)

        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)

        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop
        )

    def forward(self, x, attn_mask=None):
        x = self.attn(self.norm1(x), mask=attn_mask)
        x = x + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
