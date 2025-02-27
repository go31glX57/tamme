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

    def forward(self, inputs, recon, padding_mask, selected_mask, return_extras=False):
        # (B, S)
        mask = (padding_mask * selected_mask).bool()
        n_masked = mask.sum().clamp(1)
        # (B, *)
        mask_categoricals, value_mask = mask.split([inputs['splits'][0], sum(inputs['splits'][1:])], 1)
        mask_type_spec = mask * (inputs['pos_idx_type_spec'].squeeze(-1) >= 0)

        # (B, *)
        loss_numerics = mse(inputs['numerics'], recon['numerics'])
        loss_images = mse(inputs['images'], recon['images'])
        loss_texts = mse(inputs['texts'], recon['texts'])

        # (B, S - n_categoricals)
        value_loss = torch.cat([loss_numerics, loss_texts, loss_images], -1)
        # ()
        value_loss = (value_loss * value_mask).sum() / value_mask.sum().clamp(1)

        # (B, S)
        type_loss_cat = self.sxe(recon['types_cat'].permute(0, 2, 1), inputs['types_cat'].long())
        type_loss_spec = mse(inputs['types_spec'], recon['types_spec'])
        # ()
        type_loss_cat = (type_loss_cat * mask).sum() / n_masked
        type_loss_spec = (type_loss_spec * mask_type_spec).sum() / mask_type_spec.sum().clamp(1)
        total_loss = self.cfg.lambda_type_cat_loss * type_loss_cat + self.cfg.lambda_type_spec_loss * type_loss_spec
        total_loss = total_loss + self.cfg.lambda_val_loss * value_loss


        if return_extras:
            mask = mask.detach()
            value_mask = value_mask.detach()
            mask_type_spec = mask_type_spec.detach()
            n_masked = n_masked.detach()

            # (B, *)
            mask_numerics, mask_images, mask_texts = value_mask.split(inputs['splits'][1:], 1)

            # (B, *)
            sim_types_spec = F.cosine_similarity(recon['types_spec'].detach(), inputs['types_spec'].detach(), -1)
            sim_numerics = F.cosine_similarity(recon['numerics'].detach(), inputs['numerics'].detach(), -1)
            sim_images = F.cosine_similarity(recon['images'].detach(), inputs['images'].detach(), -1)
            sim_texts = F.cosine_similarity(recon['texts'].detach(), inputs['texts'].detach(), -1)

            # (), can be nan in case modality is not present
            sim_numerics = (sim_numerics * mask_numerics).sum() / mask_numerics.sum()
            sim_images = (sim_images * mask_images).sum() / mask_images.sum()
            sim_texts = (sim_texts * mask_texts).sum() / mask_texts.sum()
            sim_types_spec = (sim_types_spec * mask_type_spec).sum() / mask_type_spec.sum()
            sim_all = torch.stack([sim_numerics, sim_images, sim_texts, sim_types_spec]).nanmean()

            # (B, S)
            type_cat_y = inputs['types_cat'].detach().long()
            # (*, )
            type_cat_y = type_cat_y[mask > 0]
            # (B, S, vocab)
            type_cat_act = F.softmax(recon['types_cat'].detach(), -1)
            # (*, vocab)
            type_cat_act = (type_cat_act[mask > 0])

            extras = dict(
                value_loss=value_loss.detach(),
                type_loss_cat=type_loss_cat.detach(),
                type_loss_spec=type_loss_spec.detach(),

                type_cat_y=type_cat_y,
                type_cat_activations=type_cat_act,

                similarity_numerics=sim_numerics,
                similarity_images=sim_images,
                similarity_texts=sim_texts,
                similarity_types_spec=sim_types_spec,
                similarity_all=sim_all,
            )

            return total_loss, extras

        return total_loss
