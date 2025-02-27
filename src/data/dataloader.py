import os.path
import re
from functools import partial

from torch.utils.data import DataLoader, get_worker_info

from data.dataset import IterableShuffleDataset


def worker_init_fn(worker_id, num_replicas, rank, all_shard_ids, limit_data):
    worker_info = get_worker_info()

    num_replicas = max(num_replicas, 1)
    num_workers = max(worker_info.num_workers, 1)

    workers_total = num_replicas * num_workers

    selected_shards = all_shard_ids

    if workers_total > 1:
        global_worker_id = rank * num_workers + worker_id
        selected_shards = [x for x in all_shard_ids if x % workers_total == global_worker_id]

    effective_limit_data = None
    if limit_data:
        effective_limit_data = limit_data // workers_total if limit_data > workers_total else int(
            worker_id < limit_data)

    worker_info.dataset.setup(selected_shards, effective_limit_data)


def make_dataloader(cfg, dataset, is_train, device, world_size, rank):
    data_dir = os.path.join(cfg.data, dataset.split)

    shard_ids = [
        int(re.search(r'data_(\d+)\.h5', f).group(1))
        for f in os.listdir(data_dir)
        if re.match(r'data_\d+\.h5', f)
    ]

    init_fn = partial(
        worker_init_fn,
        num_replicas=world_size,
        rank=rank,
        all_shard_ids=shard_ids,
        limit_data=cfg.limit_train_data if is_train else None
    )

    if cfg.n_workers < 1:
        dataset.setup(shard_ids)

    if is_train:
        dataset = IterableShuffleDataset(dataset, buffer_size=max(cfg.buffer_factor, 2) * cfg.batch_size)

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        sampler=None,
        num_workers=cfg.n_workers,
        worker_init_fn=init_fn,
        persistent_workers=cfg.n_workers > 0,
        pin_memory=True,
        collate_fn=dataset.collate,
        prefetch_factor=cfg.prefetch_factor if cfg.n_workers > 0 else None,
    )

    return dataloader
