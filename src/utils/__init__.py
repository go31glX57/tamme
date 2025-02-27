import socket
from types import NoneType

import torch
import wandb
from omegaconf import OmegaConf
from torch.nn.utils.rnn import pad_sequence as rnn_pad_sequence
from wandb.plot.custom_chart import plot_table


"""
******************** Computational **********************
"""


def cut(x, breaks):
    """
    Discretizes x, given some bin breaks.
    :param breaks: List of breaks, right-inclusive. Will automatically extend to -inf, inf.
    """
    for i, bound in enumerate(breaks):
        if x <= bound:
            return i
    return len(breaks)


def pad_sequence(x, *args, **kwargs):
    """
    Wrapper that replaces None values with empty tensors of appropriate shape (0, n_features)
    and unsqueezes tensors to have at least 2 dimensions
    before calling torch.nn.utils.rnn.pad_sequence.

    :return the output of torch.nn.utils.rnn.pad_sequence or None if all values in x are None or empty
    """
    if x is None or len(x) == 0:
        return None

    x_safe = [s.unsqueeze(-1) if s is not None and s.ndim == 1 else s for s in x]
    n_features = [s.size(-1) for s in x_safe if s is not None]

    if len(n_features) < 1:
        return None

    assert len(set(n_features)) == 1, f"All tensors must have the same number of features. Found {[x.shape for x in x if x is not None]}"
    n_features = n_features[0]

    x_safe = [torch.empty((0, n_features)) if s is None else s for s in x_safe]

    out = rnn_pad_sequence(x_safe, *args, **kwargs)

    if out.size(1) == 0:
        return None

    return out


def sample(x, n, probs):
    B, N = x.shape

    w = probs[x]
    w = w / w.sum(-1, keepdim=True)

    sampled_indices = torch.multinomial(w, n, replacement=False)

    full_indices = torch.arange(N, device=x.device).unsqueeze(0).expand(B, N)
    mask = torch.ones_like(full_indices, dtype=torch.bool)
    mask.scatter_(1, sampled_indices, False)
    complementary_indices = full_indices[mask].view(B, N - n)

    return sampled_indices, complementary_indices


def local_window_shuffle(x: torch.Tensor, w: int, p: float = 0.5) -> torch.Tensor:
    """
    Performs localized shuffling of a 1D tensor by dividing it into non-overlapping windows and
    randomly shuffling the elements within each window with a specified probability.

    If the tensor length is not divisible by the window size, the remaining elements at the end
    are left unshuffled.

    Args:
        x (torch.Tensor): A 1D tensor of values to be shuffled.
        w (int): Size of the non-overlapping windows used for local shuffling.
        p (float): Probability of shuffling a given window. Must be between 0 and 1.

    Returns:
        torch.Tensor: A 1D tensor with locally shuffled values, with any remainder elements preserved.

    """
    n = x.size(0)

    if n < w:
        return x

    full_len = (n // w) * w  # maximum length that fits full windows
    x_main = x[:full_len]
    x_remain = x[full_len:]  # remainder (if any)

    m = x_main.unfold(0, w, w)  # shape: (num_windows, window_size)
    num_windows = m.size(0)

    mask = (torch.rand(num_windows) < p)  # mask for which windows to shuffle
    rand = torch.rand_like(m, dtype=torch.float)
    perm = rand.argsort(dim=-1)

    shuffled_windows = m.clone()
    shuffled_windows[mask] = torch.gather(m[mask], 1, perm[mask])

    shuffled = shuffled_windows.reshape(-1)

    return torch.cat([shuffled, x_remain])


def replace_col(x, col, val):
    """Along last dim """
    return torch.cat([x[..., :col], val.unsqueeze(-1), x[..., col + 1:]], dim=-1)


"""
******************** Logging **********************
"""


def matrix_to_table(x):
    """
    Takes a 2d tensor and converts it to a tensor where each row contains indices and value of one cell.
    Example:
        Input:
            tensor([[2, 0, 0],
                [0, 1, 1],
                [1, 1, 1]], dtype=torch.int32)
        Output:
            tensor([[0, 0, 2],
            [0, 1, 0],
            [0, 2, 0],
            [1, 0, 0],
            [1, 1, 1],
            [1, 2, 1],
            [2, 0, 1],
            [2, 1, 1],
            [2, 2, 1]])
    """
    assert x.dim() == 2

    rows, cols = torch.meshgrid(torch.arange(x.size(0), device=x.device),
                                torch.arange(x.size(1), device=x.device), indexing="ij")
    result = torch.stack([rows.flatten(), cols.flatten(), x.flatten()], dim=1)

    return result


def confmat_to_wandb(x, title='Confusion'):
    """
    x of shape (n_classes, n_classes) for single-label
    and (n_labels, 2, 2) for binary multi-label
    """
    x = matrix_to_table(x).cpu().tolist()

    return plot_table(
        data_table=wandb.Table(
            columns=["Actual", "Predicted", "nPredictions"],
            data=x,
        ),
        vega_spec_name="wandb/confusion_matrix/v1",
        fields={
            "Actual": "Actual",
            "Predicted": "Predicted",
            "nPredictions": "nPredictions",
        },
        string_fields={"title": title},
        split_table=False,
    )


def roc_to_wandb(x, title='ROC Curve'):
    """
    x = fpr, tpr, thresholds with fpr and tpr being (n_classes, n_thresholds+1), thresholds being 1d (n_thresholds,)
    """
    fpr, tpr, thresholds = x
    n_classes, n_thresholds = fpr.shape if fpr.ndim > 1 else (1, fpr.size(0))

    class_indices = torch.arange(n_classes, device=fpr.device, dtype=fpr.dtype).repeat_interleave(n_thresholds, -1)

    result = torch.stack([class_indices, fpr.flatten(), tpr.flatten()], -1)
    result = result.cpu().tolist()

    return plot_table(
        data_table=wandb.Table(data=result, columns=['class', 'fpr', 'tpr']),
        vega_spec_name="wandb/area-under-curve/v0",
        fields={
            "x": "fpr",
            "y": "tpr",
            "class": "class",
        },
        string_fields={
            "title": title,
            "x-axis-title": "False Positive Rate",
            "y-axis-title": "True Positive Rate",
        },
        split_table=False,
    )


def safe_metrics(data):
    """
    Filters only metrics values (from a dict) that are safe to log and display, i.e. no charts, images, etc.
    """
    res = {}

    for k, v in data.items():
        if type(v) in [float, int, str, bool, NoneType]:
            res[k] = v
        elif isinstance(v, torch.Tensor) and v.nelement() == 1:
            res[k] = v.item()

    return res


def safe_log(data):
    """
    Checks for items requiring CustomCharts nested inside a dict (currently only up to depth 1) an "un-nests" them
    before passing them to wandb.
    """

    def _process(d, prefix=None):
        res = {}
        to_top = {}
        for k, v in d.items():
            new_prefix = f'{prefix}.{k}' if prefix else k

            if isinstance(v, dict):
                r2, t2 = _process(v, prefix=new_prefix)
                res[k] = r2
                to_top.update(t2)
            elif 'confusion' in k:  # CustomCharts to top level to make wandb happy
                if v.ndim == 3:  # multilabel. shape (n_label, 2, 2)
                    for i in range(v.size(0)):
                        to_top[f'{new_prefix}_label_{i}'] = confmat_to_wandb(v[i],
                                                                             title=f'Confusion {new_prefix.title()}, Label {i}')
                else:
                    to_top[new_prefix] = confmat_to_wandb(v, title=f'Confusion {new_prefix.title()}')
            elif k == 'roc':  # CustomCharts to top level to make wandb happy
                to_top[new_prefix] = roc_to_wandb(v, title=f'ROC {new_prefix.title()}')
            elif isinstance(v, torch.Tensor):
                res[k] = v.cpu().item()
            else:
                res[k] = v

        return res, to_top

    out, extras = _process(data)

    out.update(extras)

    wandb.log(out)


"""
******************** Setup **********************
"""


def make_scheduler(cfg, optimizer, steps_per_epoch):
    match cfg.scheduler:
        case 'constant':
            return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
        case 'one-cycle':
            return torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=cfg.lr,
                epochs=cfg.n_epochs,
                steps_per_epoch=steps_per_epoch
            )
        case 'delayed-exp-decay':
            min_ratio = 0.1

            def lr_lambda(step):
                epoch = step // steps_per_epoch
                if epoch < 25:
                    return 1.0
                else:
                    decay = 0.95 ** (epoch - 25)
                    return min_ratio + (1.0 - min_ratio) * decay

            return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        case _:
            raise ValueError("Unknown scheduler: ", cfg.scheduler)


def init_wandb(cfg):
    return wandb.init(
        project=cfg.project,
        settings=wandb.Settings(start_method='thread', mode=cfg.wandb_mode),
        reinit=True,
        config=OmegaConf.to_object(cfg),
        save_code=False,
        job_type=cfg.job_type,
        group=cfg.job_group,
        dir=cfg.output_dir
    )


def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]
