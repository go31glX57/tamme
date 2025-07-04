import os
import random

import h5py
import torch

from utils import cut, pad_sequence


class IterableShuffleDataset(torch.utils.data.IterableDataset):
    def __init__(self, dataset, buffer_size):
        self.ds = dataset
        self.buffer_size = buffer_size

    def __iter__(self):
        data_iter = iter(self.ds)
        buffer = []

        try:
            for _ in range(self.buffer_size):
                buffer.append(next(data_iter))
        except StopIteration:
            pass

        while buffer:
            index = random.randint(0, len(buffer) - 1)
            yield buffer.pop(index)

            try:
                buffer.append(next(data_iter))
            except StopIteration:
                continue

    def collate(self, *args, **kwargs):
        return self.ds.collate(*args, **kwargs)

    def setup(self, *args, **kwargs):
        return self.ds.setup(*args, **kwargs)


class _BaseDataset(torch.utils.data.IterableDataset):
    def __init__(self, cfg, split, *args, **kwargs):
        super(_BaseDataset, self).__init__()
        self.cfg = cfg
        self.split = split
        self.is_training = split == "train"

        self.files = None
        self.limit_data = None

    def setup(self, shard_ids, limit_data):
        self.files = []

        for i in shard_ids:
            f = h5py.File(os.path.join(self.cfg.data, self.split, f"data_{i}.h5"))
            keys = list(f.keys())

            if self.is_training:
                random.shuffle(keys)  # remove key shuffling in case this slows down data loading

            self.files.append((f, keys))

        self.limit_data = limit_data

    def __iter__(self):
        did_one_cycle = False

        # loop infinitely for training set, but do only one cycle for val/test
        while self.is_training or (not did_one_cycle):
            if self.is_training and not self.limit_data:
                random.shuffle(self.files)

            i = 0
            for f, keys in self.files:
                for k in keys:
                    if self.is_training and self.limit_data is not None and i >= self.limit_data:
                        break

                    x = f[k]

                    ehr = {
                        col: torch.tensor(x[col][:], dtype=torch.float32) if col in x.keys() else None
                        for col in [
                            'types_categories',
                            'types_specifics',
                            'times',
                            'numerics',
                            'images',
                            'images_pos_idx',
                            'texts',
                            'texts_pos_idx'
                        ]
                    }

                    if 'meta_data' in x:
                        ehr['_meta'] = dict(zip(x['meta_columns'][:].astype('str'), x['meta_data'][:].astype('str')))

                    i += 1

                    yield ehr

                if self.is_training and self.limit_data is not None and i >= self.limit_data:
                    break

            did_one_cycle = True

    @staticmethod
    def collate(list_of_dicts):
        dict_of_lists = {key: [d[key] for d in list_of_dicts] for key in list_of_dicts[0]}

        x = {
            k: pad_sequence(v, True, -1)
            for k, v in dict_of_lists.items()
            if not k.startswith('_')
        }

        return x


class _ClassificationDataset(_BaseDataset):
    """
    Wraps a dataset and discretizes one column to be used as a target for classification.
    Target bin bounds should be right inclusive. Will extend to -inf, inf automatically.
    """

    def __init__(self, target_col: str, cfg, split, *args, **kwargs):
        super().__init__(cfg=cfg, split=split, *args, **kwargs)

        self.target_col = target_col
        self.bins = cfg.target_bins

    def __iter__(self):
        for ehr in super().__iter__():
            target = torch.tensor(cut(float(ehr['_meta'][self.target_col]), self.bins), dtype=torch.float32)
            yield ehr, target

    @staticmethod
    def collate(batch):
        ehrs, targets = zip(*batch)
        x = super(_ClassificationDataset, _ClassificationDataset).collate(ehrs)
        return x, torch.stack(targets)


class PretrainingDataset(_BaseDataset):
    pass


class HospitalStayDurationDataset(_ClassificationDataset):
    def __init__(self, cfg, split, *args, **kwargs):
        super().__init__(target_col="hosp_duration_hours", cfg=cfg, split=split, *args, **kwargs)
        assert not self.cfg.multilabel


class ICUStayDurationDataset(_ClassificationDataset):
    def __init__(self, cfg, split, *args, **kwargs):
        super().__init__(target_col="icu_duration_hours", cfg=cfg, split=split, *args, **kwargs)
        assert not self.cfg.multilabel


class ReadmissionDataset(_ClassificationDataset):
    def __init__(self, cfg, split, *args, **kwargs):
        super().__init__(target_col="readm_duration_hours", cfg=cfg, split=split, *args, **kwargs)
        assert not self.cfg.multilabel


class SurvivalDataset(_ClassificationDataset):
    def __init__(self, cfg, split, *args, **kwargs):
        super().__init__(target_col="survived", cfg=cfg, split=split, *args, **kwargs)
        assert (not self.cfg.multilabel) and len(self.cfg.target_bins) == 1


class TriageAcuityDataset(_ClassificationDataset):
    def __init__(self, cfg, split, *args, **kwargs):
        super().__init__(target_col='triage_acuity', cfg=cfg, split=split, *args, **kwargs)
        assert not self.cfg.multilabel
