from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, Normalize, Resize, ToTensor

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
    # Prepare datasets and the model
    transform = Compose([ToTensor(), Resize(32), Normalize(0.5, 0.5)])

    train = MNIST('../datasets', train=True, transform=transform, download=True)
    train_loader = DataLoader(train, batch_size=16, shuffle=True)

    val = MNIST('../datasets', train=False, transform=transform, download=True)
    val_loader = DataLoader(val, batch_size=64, shuffle=False)

    data_loaders = dict(train=train_loader, val=val_loader)
    # model = LeNet()
    net = LeNet()
    model = Mymodel(net)

    # Initialize and launch engine
    engine = Engine(model, data_loaders, stages, work_dir='../work_dirs')
    engine.launch()


if __name__ == '__main__':
    main()
