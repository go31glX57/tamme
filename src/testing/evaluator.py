import csv

import h5py
import torch
from torchmetrics import MetricCollection, Accuracy, F1Score, MatthewsCorrCoef, ConfusionMatrix, AUROC, ROC, \
    MeanMetric
from torchmetrics.functional import accuracy, f1_score
from tqdm import tqdm

from testing.metrics import BalancedAccuracy
from training.losses import MultimodalReconLoss


class BaseEvaluator:
    def __init__(self, cfg, rank, world_size, device, model, dataloader):
        """
        Evaluator class for evaluating a trained model

        Args:
            cfg (dict): Configuration dictionary.
            model (nn.Module): The trained model.
            dataloader (DataLoader): DataLoader for the test/validation dataset.
            device (torch.device): Device to run the test on.
            metric_fns (dict): Dictionary of metric functions to evaluate.
        """
        self.cfg = cfg
        self.rank = rank
        self.world_size = world_size
        self.device = device

        self.effective_steps_per_eval = None
        if self.cfg.steps_per_eval is not None:
            self.effective_steps_per_eval = self.cfg.steps_per_eval // max(self.world_size, 1)

        self.model = model
        self.dataloader = dataloader

        self.pred_writer = None
        if p := cfg.get("save_pred"):
            self._pred_file = open(p, "a", newline="")
            self.pred_writer = csv.writer(self._pred_file)

    def update_model(self, model):
        self.model = model

    def load_model(self, ckpt_path, strict=True):
        checkpoint = torch.load(
            ckpt_path,
            map_location=self.device,
            weights_only=True
        )

        m = self.model.module if self.world_size > 0 else self.model
        m.load_state_dict(checkpoint['model'], strict=strict)

    def eval_step(self, *args, **kwargs):
        raise NotImplementedError()

    def is_batch_emtpy(self, batch):
        return False

    def run(self, desc='Testing'):
        self.model.eval()

        batch_idx = 1

        pbar = tqdm(
            self.dataloader,
            desc=desc,
            leave=False,
            position=1,
            disable=self.rank > 0
        )

        dl_iter = iter(self.dataloader)

        with torch.inference_mode():
            try:
                while (self.effective_steps_per_eval is None) or batch_idx < (self.effective_steps_per_eval + 1):
                    batch = next(dl_iter)

                    if not self.is_batch_emtpy(batch):
                        self.eval_step(batch)
                        batch_idx += 1
                        pbar.update(1)

            except StopIteration:
                pass

        return None

    def __del__(self):
        if self.pred_writer:
            self._pred_file.close()


class ClassifierEvaluator(BaseEvaluator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.is_binary = self.cfg.n_classes <= 2

        task = 'multilabel' if self.cfg.multilabel else ('binary' if self.is_binary else 'multiclass')

        self.activation = torch.nn.Sigmoid() if self.is_binary else torch.nn.Softmax(-1)

        self.metrics = MetricCollection(dict(
            micro_accuracy=Accuracy(
                task=task,
                num_classes=self.cfg.n_classes,
                num_labels=self.cfg.n_multilabel,
                compute_with_cache=False,
                average='micro'
            ),
            macro_accuracy=Accuracy(
                task=task,
                num_classes=self.cfg.n_classes,
                num_labels=self.cfg.n_multilabel,
                compute_with_cache=False,
                average='macro'
            ),
            balanced_accuracy=BalancedAccuracy(
                task=task,
            ),
            f1_score=F1Score(
                task=task,
                num_classes=self.cfg.n_classes,
                num_labels=self.cfg.n_multilabel,
                compute_with_cache=False
            ),
            mcc=MatthewsCorrCoef(
                task=task,
                num_classes=self.cfg.n_classes,
                num_labels=self.cfg.n_multilabel,
                compute_with_cache=False
            ),
            confusion=ConfusionMatrix(
                task=task,
                num_classes=self.cfg.n_classes,
                num_labels=self.cfg.n_multilabel,
                compute_with_cache=False
            ),
            auroc=AUROC(
                task=task,
                num_classes=self.cfg.n_classes,
                num_labels=self.cfg.n_multilabel,
                compute_with_cache=False,
                thresholds=35
            ),
            roc=ROC(
                task=task,
                num_classes=self.cfg.n_classes,
                thresholds=35,
                num_labels=self.cfg.n_multilabel,
                compute_with_cache=False
            ),
        )).to(self.device)


    def eval_step(self, batch):
        x, y = batch

        logits = self.model(x)

        if self.cfg.n_multilabel:
            logits = logits.unsqueeze(-1)

        activations = self.activation(logits)
        y = y.to(self.device).long()

        if self.is_binary:
            y = y.unsqueeze(-1)

        self.metrics.update(activations, y)

        if w := self.pred_writer:
            y_pred = activations.squeeze(-1) if self.is_binary else activations.argmax(-1)
            w.writerows(zip(x.get('id').squeeze().tolist(), y.squeeze(-1).tolist(), y_pred.tolist(-1)))

        return None

    def run(self, *args, **kwargs):
        self.metrics.reset()

        super().run(*args, **kwargs)

        output = self.metrics.compute()

        return output

    def is_batch_emtpy(self, batch):
        return batch is None


class MAEEvaluator(BaseEvaluator):
    def __init__(self, cfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)

        self.criterion = MultimodalReconLoss(cfg).to(self.device)

        self.similarity_numerics = MeanMetric(compute_with_cache=False).to(self.device)
        self.similarity_texts = MeanMetric(compute_with_cache=False).to(self.device)
        self.similarity_images = MeanMetric(compute_with_cache=False).to(self.device)
        self.similarity_all = MeanMetric(compute_with_cache=False).to(self.device)
        self.similarity_types_spec = MeanMetric(compute_with_cache=False).to(self.device)

        self.types_cat_f1_score = MeanMetric(compute_with_cache=False).to(self.device)
        self.types_cat_macro_accuracy = MeanMetric(compute_with_cache=False).to(self.device)


    def eval_step(self, batch):
        inputs, recon, padding_mask, mask = self.model(batch)

        loss, stats = self.criterion(inputs, recon, padding_mask, mask, return_extras=True)
        loss.detach()
        del loss

        if not stats['similarity_numerics'].isnan():
            self.similarity_numerics.update(stats['similarity_numerics'])
        if not stats['similarity_texts'].isnan():
            self.similarity_texts.update(stats['similarity_texts'])
        if not stats['similarity_images'].isnan():
            self.similarity_images.update(stats['similarity_images'])
        if not stats['similarity_types_spec'].isnan():
            self.similarity_types_spec.update(stats['similarity_types_spec'])

        self.similarity_all.update(stats['similarity_all'])

        self.types_cat_macro_accuracy.update(accuracy(stats['type_cat_activations'], stats['type_cat_y'], task='multiclass', num_classes=self.cfg.type_category_vocab_size, average='macro'))
        self.types_cat_f1_score.update(f1_score(stats['type_cat_activations'], stats['type_cat_y'], task='multiclass', num_classes=self.cfg.type_category_vocab_size))

        return None

    def run(self, *args, **kwargs):
        self.similarity_numerics.reset()
        self.similarity_texts.reset()
        self.similarity_images.reset()
        self.similarity_all.reset()
        self.similarity_types_spec.reset()

        self.types_cat_macro_accuracy.reset()
        self.types_cat_f1_score.reset()

        super().run(*args, **kwargs)

        out = dict(
            similarity_numerics=self.similarity_numerics.compute(),
            similarity_texts=self.similarity_texts.compute(),
            similarity_images=self.similarity_images.compute(),
            similarity_all=self.similarity_all.compute(),
            similarity_types_spec=self.similarity_types_spec.compute(),
            types_cat_macro_accuracy=self.types_cat_macro_accuracy.compute(),
            types_cat_f1_score=self.types_cat_f1_score.compute(),
        )

        return out
