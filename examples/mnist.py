import argparse
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, Normalize, Resize, ToTensor
import nncore
from nncore.engine import Engine, comm, set_random_seed
from pydl.models.classification_model import Mymodel
from pydl.networks.classification_network import LeNet

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


def main():
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

    # Prepare datasets and the model
    transform = Compose([ToTensor(), Resize(32), Normalize(0.5, 0.5)])
    train = MNIST('../datasets', train=True, transform=transform, download=True)
    train_loader = DataLoader(train, batch_size=16, shuffle=True)
    val = MNIST('../datasets', train=False, transform=transform, download=True)
    val_loader = DataLoader(val, batch_size=64, shuffle=False)

    data_loaders = dict(train=train_loader, val=val_loader)
    net = LeNet()
    model = Mymodel(net)
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


if __name__ == '__main__':
    main()
