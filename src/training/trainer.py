import os
import sys

import torch
import torch.distributed as dist
import wandb
from omegaconf import open_dict
from torch import nn
from torchmetrics import MetricCollection, ROC, Accuracy, F1Score, MatthewsCorrCoef, ConfusionMatrix, AUROC, \
    MeanMetric
from torchmetrics.functional import accuracy, f1_score
from tqdm import tqdm

import testing
from testing.metrics import BalancedAccuracy
from training.losses import MultimodalReconLoss
from utils import safe_log, safe_metrics, make_scheduler


class BaseTrainer:
    def __init__(self, cfg, rank, world_size, device, model, train_dl, val_dl):
        self.cfg = cfg
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.effective_steps_per_epoch = self.cfg.steps_per_epoch // max(self.world_size, 1)

        self.train_dl = train_dl
        self.train_iter = None

        self.model = model

        self.optimizer = torch.optim.AdamW(
            self.get_param_groups(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay
        )

        self.scheduler = make_scheduler(cfg, self.optimizer, self.effective_steps_per_epoch)
        self.scaler = torch.GradScaler()

        self.best_epoch, self.best_result = None, None
        self.start_epoch = 1

        self.validator = getattr(testing, cfg.evaluator)(cfg, rank, world_size, device, model, val_dl)

        if rank == 0:
            os.makedirs(os.path.join(cfg.output_dir, cfg.job_group), exist_ok=True)
            run_name = [wandb.run.name]
        else:
            run_name = [None]

        if world_size > 0:
            dist.broadcast_object_list(run_name, src=0, device=device)

        self.run_name = run_name[0]

    def train_step(self, *args, **kwargs):
        raise NotImplementedError()

    def on_backward(self):
        pass

    def backward(self, loss):
        self.scaler.scale(loss).backward()

        self.scaler.unscale_(self.optimizer)
        self.on_backward()
        grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.gradclip).detach()

        self.scaler.step(self.optimizer)
        self.scheduler.step()
        self.scaler.update()

        return grad_norm

    def is_batch_emtpy(self, batch):
        return False

    def train_epoch(self):
        self.model.train()

        train_loss, train_grad_norm = 0, 0
        batch_idx = 1

        pbar = tqdm(
            list(range(self.effective_steps_per_epoch)),
            desc='Training',
            leave=False,
            position=1,
            disable=self.rank > 0
        )

        for _ in pbar:
            batch = next(self.train_iter)

            if not self.is_batch_emtpy(batch):
                self.optimizer.zero_grad(set_to_none=True)

                loss = self.train_step(batch)
                grad_norm = self.backward(loss)
                loss = loss.detach()

                if self.world_size > 0:
                    dist.reduce(loss, dst=0, op=dist.ReduceOp.AVG)
                    dist.reduce(grad_norm, dst=0, op=dist.ReduceOp.AVG)

                train_loss += loss.item()
                train_grad_norm += grad_norm

                pbar.set_postfix(dict(loss=train_loss / batch_idx))
                batch_idx += 1

        return dict(
            loss=train_loss / batch_idx,
            grad_norm_before_clip=train_grad_norm / batch_idx,
            lr=self.scheduler.get_last_lr()[0],
        )

    def run(self):
        self.best_epoch = None
        self.best_result = None
        self.start_epoch = 1

        if self.cfg.continue_from is not None:
            self.load_pretrained()

        self.train_iter = iter(self.train_dl)

        pbar_epochs = tqdm(
            range(self.start_epoch, self.cfg.n_epochs + 1),
            desc='Epochs',
            leave=False,
            position=0,
            disable=self.rank > 0
        )

        for epoch in pbar_epochs:
            if hasattr(self.train_dl.sampler, 'set_epoch'):
                self.train_dl.sampler.set_epoch(epoch)

            train_outputs = self.train_epoch()

            self.validator.update_model(self.model)
            val_outputs = self.validator.run(desc='Validating')

            if self.rank == 0:
                safe_log(dict(train=train_outputs, val=val_outputs, epoch=epoch))

                is_best = self.save_if_best(val_outputs, epoch)

                pbar_epochs.set_postfix({**safe_metrics(val_outputs), 'best_epoch': self.best_epoch})

                if not is_best:
                    self.save_if_it_is_time(epoch)

    def load_pretrained(self):
        weights_only = not self.cfg.restore_training_state

        checkpoint = torch.load(self.cfg.continue_from, map_location=self.device, weights_only=weights_only)

        m = self.model.module if self.world_size > 0 else self.model
        k_missing, k_unexpected = m.load_state_dict(checkpoint['model'], strict=not weights_only)

        if self.rank == 0:
            print('Loaded weights: ', self.cfg.continue_from, file=sys.stderr)
            print('Missing keys: ', k_missing, file=sys.stderr)
            print('Unexpected keys: ', k_unexpected, file=sys.stderr)

        if not weights_only:
            self.start_epoch = checkpoint['epoch'] + 1
            self.best_epoch = checkpoint['best_epoch']
            self.best_result = checkpoint['best_result']
            self.scheduler.load_state_dict(checkpoint['scheduler'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])

            if self.rank == 0:
                print('Loaded training state: ', self.cfg.continue_from, file=sys.stderr)

    def save_if_best(self, val_metrics, epoch):
        res = val_metrics[self.cfg.monitor]

        if self.best_result is None or res > self.best_result:
            self.best_epoch, self.best_result = epoch, res

            wandb.run.summary["best_epoch"] = self.best_epoch
            wandb.run.summary["best_result"] = self.best_result

            self.save(epoch, 'best')
            return True

        return False

    def save_if_it_is_time(self, epoch):
        make_regular_checkpoint = (epoch % self.cfg.checkpoint_every == 0) or epoch == self.cfg.n_epochs

        if make_regular_checkpoint:
            self.save(epoch, 'latest')
            return True

        return False

    @property
    def checkpoint_file_base(self):
        run_id = self.run_name.strip().replace(' ', '_')
        return os.path.join(self.cfg.output_dir, self.cfg.job_group, run_id)

    def save(self, epoch, suffix: str = ""):
        m = self.model.module if self.world_size > 0 else self.model

        checkpoint = dict(
            epoch=epoch,
            model=m.state_dict(),
            optimizer=self.optimizer.state_dict(),
            scheduler=self.scheduler.state_dict(),
            best_epoch=self.best_epoch,
            best_result=self.best_result
        )

        torch.save(checkpoint, f'{self.checkpoint_file_base}_{suffix}.pt')

    def get_param_groups(self):
        """Setup optimizer with different learning rates per layer if specified"""
        factor = getattr(self.cfg, 'lr_multiplier', 1)

        m = self.model.module if self.world_size > 0 else self.model
        trainable_params, frozen_params = m.get_trainable_parameters()

        param_groups = []

        if frozen_params:
            param_groups.append({'params': frozen_params, 'lr': 0 })

        if trainable_params:
            param_groups.append({ 'params': trainable_params, 'lr': self.cfg.lr * factor})

        return param_groups


class ClassifierTrainer(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.is_binary = self.cfg.n_classes <= 2

        task = 'multilabel' if self.cfg.multilabel else ('binary' if self.is_binary else 'multiclass')

        w = None
        if self.cfg.cls_weights is not None:
            w = torch.tensor(self.cfg.cls_weights, device=self.device, dtype=torch.float)

        if self.is_binary:
            loss = nn.BCEWithLogitsLoss(pos_weight=w)
            self.activation = torch.nn.Sigmoid()
        else:
            loss = nn.CrossEntropyLoss(weight=w, label_smoothing=self.cfg.label_smoothing)
            self.activation = torch.nn.Softmax(-1)

        self.criterion = loss.to(self.device)

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


        self._saved_step_inputs = None


    def train_step(self, batch):
        x, y = batch

        y = y.to(self.device)

        logits = self.model(x)

        if self.cfg.n_multilabel:
            logits = logits.unsqueeze(-1)

        loss = self.criterion(logits, y.unsqueeze(-1).float() if self.is_binary else y.long())

        activations = self.activation(logits).detach()
        y = y.detach().long()

        if self.is_binary:
            y = y.unsqueeze(-1)

        self.metrics.update(activations, y)

        if self.cfg.track_attention:
            self._saved_step_inputs = x

        return loss

    def train_epoch(self, *args, **kwargs):
        self.metrics.reset()

        output = super().train_epoch(*args, **kwargs)

        output.update(self.metrics.compute())

        return output

    def is_batch_emtpy(self, batch):
        return batch is None or batch[0]['types_categories'].size(1) <= 0


class MAETrainer(BaseTrainer):
    def __init__(self, cfg, *args, **kwargs):

        with open_dict(cfg):
            cfg['emb_dropout'] = 0  # Force disable emb dropout for masked sequence modelling
            cfg['pos_jitter_prob'] = 0  # Force disable pos jitter for masked sequence modelling

        super().__init__(cfg, *args, **kwargs)

        self.criterion = MultimodalReconLoss(cfg).to(self.device)

        # don't use torchmetrics.CosineSimilarity since it will store ALL inputs till compute
        self.similarity_numerics = MeanMetric(compute_with_cache=False).to(self.device)
        self.similarity_texts = MeanMetric(compute_with_cache=False).to(self.device)
        self.similarity_images = MeanMetric(compute_with_cache=False).to(self.device)
        self.similarity_all = MeanMetric(compute_with_cache=False).to(self.device)
        self.similarity_types_spec = MeanMetric(compute_with_cache=False).to(self.device)

        self.types_cat_f1_score = MeanMetric(compute_with_cache=False).to(self.device)
        self.types_cat_macro_accuracy = MeanMetric(compute_with_cache=False).to(self.device)

        self.mean_value_loss = MeanMetric(compute_with_cache=False).to(self.device)
        self.mean_type_loss_cat = MeanMetric(compute_with_cache=False).to(self.device)
        self.mean_type_loss_spec = MeanMetric(compute_with_cache=False).to(self.device)

    def train_step(self, batch):
        inputs, recon, padding_mask, mask = self.model(batch)

        loss, stats = self.criterion(inputs, recon, padding_mask, mask, return_extras=True)

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

        self.mean_type_loss_cat.update(stats['type_loss_cat'])
        self.mean_type_loss_spec.update(stats['type_loss_spec'])
        self.mean_value_loss.update(stats['value_loss'])

        return loss

    def train_epoch(self, *args, **kwargs):
        self.similarity_numerics.reset()
        self.similarity_texts.reset()
        self.similarity_images.reset()
        self.similarity_all.reset()
        self.similarity_types_spec.reset()

        self.types_cat_macro_accuracy.reset()
        self.types_cat_f1_score.reset()

        self.mean_value_loss.reset()
        self.mean_type_loss_cat.reset()
        self.mean_type_loss_spec.reset()

        out = super().train_epoch(*args, **kwargs)

        out.update(dict(
            similarity_numerics=self.similarity_numerics.compute(),
            similarity_texts=self.similarity_texts.compute(),
            similarity_images=self.similarity_images.compute(),
            similarity_all=self.similarity_all.compute(),
            similarity_types_spec=self.similarity_types_spec.compute(),

            types_cat_macro_accuracy=self.types_cat_macro_accuracy.compute(),
            types_cat_f1_score=self.types_cat_f1_score.compute(),

            value_loss=self.mean_value_loss.compute(),
            type_loss_cat=self.mean_type_loss_cat.compute(),
            type_loss_spect=self.mean_type_loss_spec.compute(),
        ))

        return out
