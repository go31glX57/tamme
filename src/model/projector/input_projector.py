import torch
from timm.layers import trunc_normal_
from torch import nn


class FourierFeatures(nn.Module):
    def __init__(self, cfg):
        super(FourierFeatures, self).__init__()
        assert len(cfg.fourier_scales) > 0

        _omega = 1 / (2 ** torch.tensor(cfg.fourier_scales, dtype=torch.float32))
        self.register_buffer('omega', _omega)

        self.output_dim = 2 * len(self.omega)

    def forward(self, x):
        x = torch.einsum('bs,d -> bsd', [x.squeeze(-1), self.omega])
        x_sin = torch.sin(x)
        x_cos = torch.cos(x)
        res = torch.cat((x_sin, x_cos), -1)
        return res


class MultiModalInputProjector(nn.Module):
    def __init__(self, cfg, return_inputs: bool = False):
        super(MultiModalInputProjector, self).__init__()

        self.cfg = cfg
        self.return_inputs = return_inputs

        self.embed_type_category = nn.Embedding(cfg.type_category_vocab_size + 1, cfg.emb_dim, padding_idx=0)
        embed_type_category_weight = torch.nn.Parameter(torch.randn_like(self.embed_type_category.weight) * .02)
        trunc_normal_(embed_type_category_weight, std=.02)
        with torch.no_grad():
            self.embed_type_category.weight[1:] = embed_type_category_weight[1:]

        self.encoder_numeric = FourierFeatures(cfg)

        self.proj_type_specifics = nn.Linear(cfg.emb_dim_type_specifics, cfg.emb_dim)
        self.proj_numeric = nn.Linear(self.encoder_numeric.output_dim, cfg.emb_dim)
        self.proj_text = nn.Linear(cfg.emb_dim_text, cfg.emb_dim)
        self.proj_image = nn.Linear(cfg.emb_dim_image, cfg.emb_dim)

    @property
    def device(self):
        return self.embed_type_category.weight.device

    def forward(self, x):
        """
        :param x: dict of modalities
        """

        # Get types.
        types_cat = x['types_categories'].to(self.device, non_blocking=True).squeeze(-1).int()
        types_spec = x['types_specifics'].to(self.device, non_blocking=True)

        # Get shape of batched sequences. Resulting seq might be longer due to padding.
        B, L = types_cat.shape

        # Get numerics.
        if x['numerics'] is not None:
            numerics = x['numerics'].to(self.device, non_blocking=True)
            pos_idx_numerics, val_numerics = numerics.split([1, 1], -1)
            val_numerics = self.encoder_numeric(val_numerics)
        else:
            val_numerics = torch.empty((B, 0, self.encoder_numeric.output_dim), dtype=torch.float32, device=self.device)
            pos_idx_numerics = torch.full((B, 0, 1), -1, dtype=torch.float32, device=self.device)

        # Get images.
        if x['images'] is not None:
            images = x['images'].to(self.device, non_blocking=True)
            pos_idx_images = x['images_pos_idx'].to(self.device, non_blocking=True)
        else:
            images = torch.empty((B, 0, self.cfg.emb_dim_image), dtype=torch.float32, device=self.device)
            pos_idx_images = torch.full((B, 0, 1), -1, dtype=torch.float32, device=self.device)

        # Get texts.
        if x['texts'] is not None:
            texts = x['texts'].to(self.device, non_blocking=True)
            pos_idx_texts = x['texts_pos_idx'].to(self.device, non_blocking=True)
        else:
            texts = torch.empty((B, 0, self.cfg.emb_dim_text), dtype=torch.float32, device=self.device)
            pos_idx_texts = torch.full((B, 0, 1), -1, dtype=torch.float32, device=self.device)

        # Process numerics.
        emb_numerics = self.proj_numeric(val_numerics)
        attn_mask_numerics = pos_idx_numerics >= 0

        # Process texts.
        emb_texts = self.proj_text(texts)
        attn_mask_texts = pos_idx_texts >= 0

        # Process images.
        emb_images = self.proj_image(images)
        attn_mask_images = pos_idx_images >= 0

        # Concat value embeddings.
        emb = torch.cat([emb_numerics, emb_images, emb_texts], 1)
        attn_mask = torch.cat([attn_mask_numerics, attn_mask_images, attn_mask_texts], 1).reshape(B, 1, 1, -1)
        pos_idx = torch.cat([pos_idx_numerics, pos_idx_images, pos_idx_texts], 1).long()

        # Resolve negative indices.
        pos_idx_res = (pos_idx + L) % L

        # Get types according to position index.
        types_cat = types_cat.gather(1, pos_idx_res.squeeze(-1))
        types_spec = types_spec.gather(1, pos_idx_res.repeat(1, 1, types_spec.size(-1)))
        emb_type_cat = self.embed_type_category(1 + types_cat)  # -1 padding -> 0
        emb_type_spec = self.proj_type_specifics(types_spec)

        # Add type emb.
        emb = emb + emb_type_cat + emb_type_spec

        # Get time according to position index.
        time = x['times'].to(self.device, non_blocking=True)
        time = time.gather(1, pos_idx_res.repeat(1, 1, time.size(-1)))

        if self.return_inputs:
            splits = [
                val_numerics.size(1),
                images.size(1),
                texts.size(1)
            ]

            inputs = dict(
                types_cat=types_cat,
                types_spec=types_spec,
                numerics=val_numerics,
                images=images,
                texts=texts,
                splits=splits
            )

            return emb, pos_idx, attn_mask, time, inputs

        return emb, pos_idx, attn_mask, time
