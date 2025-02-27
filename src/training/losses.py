import torch
import torch.nn.functional as F
from torch import nn
import polars as pl


def mse(x, y):
    loss = (x - y) ** 2
    loss = loss.mean(-1)
    return loss


class MultimodalReconLoss(nn.Module):
    def __init__(self, cfg):
        super(MultimodalReconLoss, self).__init__()
        self.cfg = cfg

        d = torch.tensor(pl.read_csv(cfg.type_cat_freq)['count'], requires_grad=False)
        w = (1 / d.clamp(1)) ** (1 - cfg.loss_weight_smoothing)
        w[d == 0] = 0
        w = w / w.sum()

        self.sxe = nn.CrossEntropyLoss(reduction='none', ignore_index=-1, weight=w, label_smoothing=cfg.label_smoothing)

    def forward(self, inputs, recon, padding_mask, selected_mask):
        # (B, S)
        mask = (padding_mask * selected_mask).bool()
        n_masked = mask.sum()
        # (B, *)
        mask_numerics, mask_images, mask_texts = mask.split(inputs['splits'], 1)

        # (B, *)
        loss_numerics = mse(inputs['numerics'], recon['numerics'])
        loss_images = mse(inputs['images'], recon['images'])
        loss_texts = mse(inputs['texts'], recon['texts'])

        # (B, S)
        value_loss = torch.cat([loss_numerics, loss_texts, loss_images], -1)
        # ()
        value_loss = (value_loss * mask).sum() / n_masked

        # (B, S)
        loss_type_spec = mse(inputs['types_spec'], recon['types_spec'])
        loss_type_cat = self.sxe(recon['types_cat'].permute(0, 2, 1), inputs['types_cat'].long())
        # ()
        type_loss = loss_type_cat + loss_type_spec
        type_loss = (type_loss * mask).sum() / n_masked

        # ()
        total_loss = self.cfg.lambda_val_loss * value_loss + self.cfg.lambda_type_loss * type_loss

        mask = mask.detach()
        mask_numerics, mask_images, mask_texts = mask_numerics.detach(), mask_images.detach(), mask_texts.detach()
        n_masked = n_masked.detach()

        # (B, *)
        sim_numerics = F.cosine_similarity(recon['numerics'].detach(), inputs['numerics'].detach(), -1)
        sim_images = F.cosine_similarity(recon['images'].detach(), inputs['images'].detach(), -1)
        sim_texts = F.cosine_similarity(recon['texts'].detach(), inputs['texts'].detach(), -1)
        # (B, S)
        sim_types_spec = F.cosine_similarity(recon['types_spec'].detach(), inputs['types_spec'].detach(), -1)
        # (B, S)
        sim_all = torch.cat([sim_numerics, sim_images, sim_texts], -1)
        # (B, S, 2)
        sim_all = torch.stack([sim_all, sim_types_spec], -1)

        # ()
        sim_all = (sim_all * mask.unsqueeze(-1)).sum() / (n_masked * 2)
        sim_numerics = (sim_numerics * mask_numerics).sum() / mask_numerics.sum()
        sim_images = (sim_images * mask_images).sum() / mask_images.sum()
        sim_texts = (sim_texts * mask_texts).sum() / mask_texts.sum()
        sim_types_spec = (sim_types_spec * mask).sum() / n_masked

        # (B, S)
        type_cat_y = inputs['types_cat'].detach().long()
        # (*, )
        type_cat_y = type_cat_y[mask > 0]
        # (B, S, vocab)
        type_cat_act = F.softmax(recon['types_cat'].detach(), -1)
        # (*, vocab)
        type_cat_act = (type_cat_act[mask > 0])

        extras = dict(
            val_loss=value_loss,
            type_loss=type_loss,
            type_cat_y=type_cat_y,
            type_cat_activations=type_cat_act,
            similarity_numerics=sim_numerics,
            similarity_images=sim_images,
            similarity_texts=sim_texts,
            similarity_all=sim_all,
            similarity_types_spec=sim_types_spec,
        )

        return total_loss, extras
