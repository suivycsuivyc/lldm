"""
    Parse args and make cfg, then spawn trainers which run according to cfg.
"""
import argparse
import os

def make_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default='cfgs/_.yaml')
    parser.add_argument('--opt', nargs='*', default=[])
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--resume', '-r', action='store_true')
    parser.add_argument('--force-replace', '-f', action='store_true')

    parser.add_argument('--gpu', '-g')
    parser.add_argument('--port', default=None)

    parser.add_argument('--save-root', default='save')
    parser.add_argument('--name', '-n', default=None)
    parser.add_argument('--tag', default=None)
    parser.add_argument('--wandb', '-w', action='store_true')
    parser.add_argument('--verbose', '-v', action='store_true')
    args = parser.parse_args()
    return args

args = make_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
world_size = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
os.environ['WORLD_SIZE'] = str(world_size)

import torch.multiprocessing as mp
from omegaconf import OmegaConf

from trainers import trainers_dict
from utils import ensure_path


def parse_cfg(cfg):
    if cfg.get('_base_') is not None:
        fnames = cfg.pop('_base_')
        if isinstance(fnames, str):
            fnames = [fnames]
        base_cfg = OmegaConf.merge(*[parse_cfg(OmegaConf.load(_)) for _ in fnames])
        cfg = OmegaConf.merge(base_cfg, cfg)
    return cfg


def make_cfg(args):
    cfg = parse_cfg(OmegaConf.load(args.cfg))
    for i in range(0, len(args.opt), 2):
        k, v = args.opt[i: i + 2]
        OmegaConf.update(cfg, k, v)
    cfg.random_seed = args.seed

    env = OmegaConf.create()
    if args.port is None:
        first_gpu_id = os.environ['CUDA_VISIBLE_DEVICES'].split(',')[0]
        env.port = str(29600 + int(first_gpu_id))
    else:
        env.port = args.port
    if args.name is None:
        exp_name = os.path.splitext(os.path.basename(args.cfg))[0]
    else:
        exp_name = args.name
    if args.tag is not None:
        exp_name += '_' + args.tag
    env.exp_name = exp_name
    env.save_dir = os.path.join(args.save_root, exp_name)
    env.wandb = args.wandb
    env.verbose = args.verbose
    cfg._env = env
    return cfg


def main_worker(rank, cfg):
    trainer = trainers_dict[cfg.trainer](rank, cfg)
    trainer.run()


if __name__ == '__main__':
    cfg = make_cfg(args)

    force_replace = False
    if args.resume:
        replace = False
    else:
        replace = True
        if args.force_replace:
            force_replace = True
    ensure_path(cfg._env.save_dir, replace=replace, force_replace=force_replace)

    if world_size > 1:
        mp.spawn(main_worker, args=(cfg,), nprocs=world_size)
    else:
        main_worker(0, cfg)
