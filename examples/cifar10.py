import argparse
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import nncore
from nncore.engine import Engine, set_random_seed, comm
import timm
from nncore.nn import build_model
from pydl.networks.lenet import LeNet


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='../configs/common/base_30e_resnet18.py', help='config file')
    parser.add_argument('--checkpoint', default='', help='load a checkpoint')
    parser.add_argument('--resume', help='resume from a checkpoint')
    parser.add_argument('--seed', help='random seed')
    parser.add_argument('--eval', help='evaluation mode', action='store_true')
    parser.add_argument('--launcher', help='job launcher')
    args = parser.parse_args()
    return args


def dataloder_cifar10():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_set = torchvision.datasets.CIFAR10(root='../datasets', train=True,
                                             download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=128,
                                               shuffle=True, num_workers=8)

    val_set = torchvision.datasets.CIFAR10(root='../datasets', train=False,
                                           download=True, transform=transform)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=5000,
                                             shuffle=False, num_workers=8)

    return dict(train=train_loader, val=val_loader)


def main():
    args = parse_args()
    cfg = nncore.Config.from_file(args.config)

    launcher = comm.init_dist()
    time_str = nncore.get_timestamp()
    work_dir = nncore.join('../work_dirs', time_str)
    log_file = nncore.join(work_dir, '{}.log'.format(time_str))
    logger = nncore.get_logger(log_file=log_file)
    logger.info(f'Environment info:\n{nncore.collect_env_info()}')
    logger.info(f'Elastic launcher: {launcher}')
    seed = args.seed if args.seed is not None else cfg.get('seed')
    seed = set_random_seed(seed=seed, deterministic=True)
    logger.info(f'Using random seed: {seed}')

    data_loaders = dataloder_cifar10()
    net = timm.create_model('resnet18', num_classes=10, pretrained=True,
                            pretrained_cfg_overlay=dict(file='../pretrained_weights/resnet18.pth'))

    model = build_model(dict(type='Mymodel', net=net), dist=bool(launcher))
    # Initialize and launch engine
    engine = Engine(
        model=model,
        data_loaders=data_loaders,
        stages=cfg.stages,
        work_dir=work_dir,
        amp=True,
        seed=seed,
        hooks=cfg.hooks,
    )
    engine.launch()


if __name__ == '__main__':
    main()
