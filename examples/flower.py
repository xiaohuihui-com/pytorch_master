import os
import json
import argparse
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import nncore
from nncore.nn import build_model
from nncore.engine import Engine, set_random_seed, comm
from pydl.models.classification_model import Mymodel
import timm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='../configs/common/base_30e_resnet18.py', help='config file')
    parser.add_argument('--checkpoint', help='load a checkpoint')
    parser.add_argument('--resume', help='resume from a checkpoint')
    parser.add_argument('--seed', help='random seed')
    parser.add_argument('--eval', help='evaluation mode', action='store_true')
    parser.add_argument('--launcher', help='job launcher')
    args = parser.parse_args()
    return args


def dataloder_flower():
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
        "val": transforms.Compose([transforms.Resize((224, 224)),  # cannot 224, must (224, 224)
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    }

    data_root = os.path.abspath(os.path.join(os.getcwd(), "../"))  # get datasets root path
    image_path = os.path.join(data_root, "datasets", "flower_data")  # flower datasets set path
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                         transform=data_transform["train"])

    flower_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in flower_list.items())
    # write dict into json file
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=64,
                                               shuffle=True,
                                               num_workers=8)

    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                            transform=data_transform["val"])
    val_loader = torch.utils.data.DataLoader(validate_dataset,
                                             batch_size=4,
                                             shuffle=False,
                                             num_workers=8)

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

    data_loaders = dataloder_flower()
    net = timm.create_model('resnet18', num_classes=5, pretrained=True,
                            pretrained_cfg_overlay=dict(file='../pretrained_weights/resnet18.pth'))
    model = build_model(dict(type='Mymodel', net=net), dist=bool(launcher))
    logger.info(model)
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
