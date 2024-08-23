import argparse
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import nncore
from nncore.engine import Engine, set_random_seed, comm
from pydl.models.classification_model import Mymodel
# from pydl.networks.classification_network import LeNet
from pydl.networks.lenet import LeNet

stages = [
    dict(
        epochs=5,
        optimizer=dict(type='Adam', lr=1e-3),
        warmup=dict(type='iter', policy='linear', steps=500, ratio=0.001)),
    dict(
        epochs=10,
        optimizer=dict(type='SGD', lr=1e-3, momentum=0.9),
        lr_schedule=dict(type='iter', policy='cosine'),
        validation=dict(interval=1))
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='config file')
    parser.add_argument('--checkpoint', help='load a checkpoint')
    parser.add_argument('--resume', help='resume from a checkpoint')
    parser.add_argument('--seed', help='random seed')
    parser.add_argument('--eval', help='evaluation mode', action='store_true')
    parser.add_argument('--launcher', help='job launcher')
    args = parser.parse_args()
    return args


def main():
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

    data_loaders = dict(train=train_loader, val=val_loader)

    net = LeNet()
    model = Mymodel(net)
    launcher = comm.init_dist(launcher=None)
    time_str = nncore.get_timestamp()
    work_dir = nncore.join('../work_dirs', time_str)
    log_file = nncore.join(work_dir, '{}.log'.format(time_str))
    logger = nncore.get_logger(log_file=log_file)
    logger.info(f'Environment info:\n{nncore.collect_env_info()}')
    logger.info(f'Elastic launcher: {launcher}')
    # seed = args.seed if args.seed is not None else cfg.get('seed')
    seed = set_random_seed(seed=None, deterministic=True)
    logger.info(f'Using random seed: {seed}')

    # Initialize and launch engine
    engine = Engine(
        model=model,
        data_loaders=data_loaders,
        stages=stages,
        work_dir=work_dir,
        amp=True,
        seed=seed
    )
    engine.launch()


if __name__ == '__main__':
    main()
