import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import nncore
from nncore.engine import Engine
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
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_set = torchvision.datasets.CIFAR10(root='../datasets', train=True,
                                             download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=36,
                                               shuffle=True, num_workers=8)

    val_set = torchvision.datasets.CIFAR10(root='../datasets', train=False,
                                           download=True, transform=transform)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=5000,
                                             shuffle=False, num_workers=8)

    data_loaders = dict(train=train_loader, val=val_loader)
    net = LeNet()
    model = Mymodel(net)
    time_str = nncore.get_timestamp()
    work_dir = nncore.join('../work_dirs', time_str)
    # Initialize and launch engine
    engine = Engine(
        model=model,
        data_loaders=data_loaders,
        stages=stages,
        work_dir=work_dir
    )
    engine.launch()


if __name__ == '__main__':
    main()
