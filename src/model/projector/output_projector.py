from torch import nn


class MultiModalOutputProjector(nn.Module):
    def __init__(self, cfg):
        super(MultiModalOutputProjector, self).__init__()
        self.cfg = cfg

        self.features_type_cat = nn.Sequential(nn.Linear(cfg.emb_dim_dec, cfg.emb_dim_dec), nn.LeakyReLU())
        self.head_type_cat = nn.Linear(cfg.emb_dim_dec, cfg.type_category_vocab_size)
        self.features_type_spec = nn.Sequential(nn.Linear(cfg.emb_dim_dec, cfg.emb_dim_dec), nn.LeakyReLU())
        self.head_type_spec = nn.Linear(cfg.emb_dim_dec, cfg.emb_dim_type_specifics)

        self.features_value = nn.Sequential(nn.Linear(cfg.emb_dim_dec, cfg.emb_dim_dec), nn.LeakyReLU())
        self.head_numeric = nn.Sequential(nn.Linear(cfg.emb_dim_dec, 2 * len(cfg.fourier_scales)), nn.Tanh())
        self.head_text = nn.Linear(cfg.emb_dim_dec, cfg.emb_dim_text)
        self.head_image = nn.Linear(cfg.emb_dim_dec, cfg.emb_dim_image)

    def forward(self, x_hat, splits):
        # x: (B, S, E_dec)
        # mask: (B, S, 1)
        # splits: list[int], length 3

        # (B, S, E_dec)
        features_type_cat = self.features_type_cat(x_hat)
        features_type_spec = self.features_type_spec(x_hat + features_type_cat)
        features_value = self.features_value(x_hat + features_type_spec)

        # (B, S, *)
        types_cat_logits = self.head_type_cat(features_type_cat)
        types_spec = self.head_type_spec(features_type_spec)

        # (B, *, E_dec)
        _, features_numerics, features_images, features_text = features_value.split(splits, 1)

        # (B, *, *)
        numerics = self.head_numeric(features_numerics)
        images = self.head_image(features_images)
        texts = self.head_text(features_text)

        return dict(types_cat=types_cat_logits, types_spec=types_spec, numerics=numerics, texts=texts, images=images)
