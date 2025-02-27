import os
from pprint import pprint

import hydra
import polars as pl
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import wandb
from omegaconf import open_dict
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

import data
import model as model_repo
import testing
import training
from data import make_dataloader
from utils import find_free_port, safe_log, safe_metrics, init_wandb


def run(rank, world_size, cfg, return_dict):
    do_train = cfg.mode == 'train'
    do_test = not (hasattr(cfg, 'omit_test') and cfg.omit_test)

    torch.multiprocessing.set_sharing_strategy('file_system')
    torch.set_float32_matmul_precision('high')

    device = torch.device(f'cuda:{rank}' if world_size > 0 else 'cpu')

    if world_size > 0:
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    if rank == 0:
        return_dict['best_val_res'] = []

    for seed in (cfg.seeds if do_train else [42]):
        torch.manual_seed(seed)

        model = getattr(model_repo, cfg.model)(cfg)
        model.to(device)

        with open_dict(cfg):
            cfg['trainable_params'] = sum(p.numel() for p in model.parameters() if p.requires_grad)

        if world_size > 0:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = DDP(model, device_ids=[rank], find_unused_parameters=True)

        if rank == 0:
            run = init_wandb(cfg)

        Dataset = getattr(data, cfg.dataset)

        if do_train:
            train_ds = Dataset(cfg, 'train')
            train_dl = make_dataloader(cfg, train_ds, True, device, world_size, rank)

            val_ds = Dataset(cfg, 'val')
            val_dl = make_dataloader(cfg, val_ds, False, device, world_size, rank)

            trainer = getattr(training, cfg.trainer)(cfg, rank, world_size, device, model, train_dl, val_dl)
            trainer.run()
        else:
            assert cfg.continue_from is not None

        test_metrics = None

        if do_test:
            test_ds = Dataset(cfg, 'test')
            test_dl = make_dataloader(cfg, test_ds, False, device, world_size, rank)

            tester = getattr(testing, cfg.evaluator)(cfg, rank, world_size, device, model, test_dl)
            _ckpt = f'{trainer.checkpoint_file_base}_best.pt' if do_train else cfg.continue_from
            tester.load_model(_ckpt, strict=do_train)

            test_metrics = tester.run()

            _m = model.module if world_size > 0 else model
            if hasattr(_m, 'finish'):
                _m.finish()

        if rank == 0:
            if test_metrics is not None:
                safe_log(dict(test=test_metrics))
                print('Test results\n', "-" * 25, )
                pprint(safe_metrics(test_metrics))
            elif do_test:
                print("Done testing. No test metrics collected.")

            if cfg.wandb_mode != 'disable':
                return_dict['best_val_res'] = return_dict['best_val_res'] + [run.summary.get('best_result')]
                wandb.finish()

        if world_size > 0:
            dist.barrier()

    if world_size > 0 and rank == 0:
        dist.destroy_process_group()


@hydra.main(config_path='./configs', version_base='1.2', config_name=None)
def main(cfg):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(find_free_port())

    num_gpus = torch.cuda.device_count()

    with open_dict(cfg):
        cfg['output_dir'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'runs')
        cfg['type_category_vocab_size'] = len(pl.read_csv(cfg.type_category_vocab))
        cfg['num_gpus'] = num_gpus
        cfg['n_classes'] = len(cfg.target_bins) + 1 if 'target_bins' in cfg else None
        cfg['multilabel'] = bool(cfg.n_multilabel) if 'n_multilabel' in cfg else False
        cfg['slurm_id'] = os.getenv('SLURM_JOB_ID')

    manager = mp.Manager()
    return_dict = manager.dict()

    if num_gpus > 0:
        mp.spawn(run, args=(num_gpus, cfg, return_dict), nprocs=num_gpus)
    else:
        run(0, 0, cfg, return_dict)

    best_val_res = return_dict.get('best_val_res', None)
    return best_val_res


if __name__ == '__main__':
    main()
