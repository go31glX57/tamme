import warnings

import torch
from sklearn.metrics import balanced_accuracy_score
from torchmetrics import Metric
from torchmetrics.utilities import dim_zero_cat


class BalancedAccuracy(Metric):
    higher_is_better = True

    def __init__(self, task):
        super().__init__()

        if task == 'multilabel':
            raise NotImplementedError()

        self.task = task
        self.is_binary = task == 'binary'

        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("targets", default=[], dist_reduce_fx="cat")

    def update(self, activations, targets):
        preds = (activations >= 0.5).int() if self.is_binary else activations.argmax(-1).int()

        self.preds.append(preds.flatten())
        self.targets.append(targets.flatten())

    def compute(self):
        y_pred = dim_zero_cat(self.preds)
        y = dim_zero_cat(self.targets)

        with warnings.catch_warnings():
            warnings.simplefilter("always")
            res = balanced_accuracy_score(y.cpu(), y_pred.cpu())

        return torch.tensor(res, device=self.device, dtype=torch.float32)
