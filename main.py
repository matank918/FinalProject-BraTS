import importlib
import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import LambdaLR
from utils.utils import get_number_of_learnable_parameters
from utils.Log import get_logger, get_module_variable, get_checkpoint_dir
from DataLoader.CustomDataset import CustomDataset
import utils.config as cfg
from monai.losses import DiceLoss, DiceCELoss
from DataLoader.BasicTransformations import train_transforms, val_transform
from monai.networks.layers.factories import Act, Norm
from trainer import UNet3DTrainer
from torch.utils.data import DataLoader, random_split
from monai.data import DataLoader
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.networks.nets import UNet
import torch
import copy
from monai.utils import set_determinism


def _get_loaders(dataset, val_percent, batch_size):
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train, val = random_split(dataset, [n_train, n_val])
    train = copy.deepcopy(train)
    val = copy.deepcopy(val)
    train.dataset.transform = train_transforms
    val.dataset.transform = val_transform
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)
    return train_loader, val_loader


def _get_model():
    return UNet(dimensions=cfg.dimensions, in_channels=cfg.in_channels, out_channels=cfg.out_channels,
                channels=cfg.f_maps,
                strides=cfg.strides, num_res_units=cfg.num_res_units, act=Act.LEAKYRELU, norm=Norm.BATCH)


def _create_loss():
    # Create loss criterion
    if cfg.loss_name == "Dice":
        return DiceLoss(to_onehot_y=False, squared_pred=True, softmax=True, include_background=True)
    if cfg.loss_name == "DiceCE":
        return DiceCELoss(to_onehot_y=False, squared_pred=True, softmax=True, include_background=True)


def _create_optimizer(model):
    # Create loss criterion
    if cfg.optimizer_name == "Adam":
        return Adam(model.parameters(), lr=cfg.initial_lr, weight_decay=cfg.weight_decay, amsgrad=cfg.amsgrad)
    if cfg.optimizer_name == "SGD":
        return SGD(model.parameters(), lr=cfg.initial_lr, nesterov=cfg.nesterov, momentum=cfg.momentum)


def _create_scheduler(optimizer):
    lambda1 = lambda epoch: 0.95 ** epoch
    return LambdaLR(optimizer, lr_lambda=lambda1)


def _create_eval():
    return DiceMetric(include_background=True, reduction="mean")


if __name__ == '__main__':
    # torch.manual_seed(0)

    # Load and log experiment configuration
    logger = get_logger(cfg.log_path)
    logger.info(cfg.id)
    logger.info(cfg.run_name)
    logger.info(cfg.run_purpose)
    logger.info(get_module_variable(cfg))

    # Create data loaders
    dataset = CustomDataset(cfg.loader_path)
    train_loader, eval_loader = _get_loaders(dataset, val_percent=cfg.val_percent, batch_size=cfg.batch_size)

    # set model
    model = _get_model()

    # use DataParallel if more than 1 GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1 and not device.type == 'cpu':
        model = nn.DataParallel(model)
        logger.info(f'Using {torch.cuda.device_count()} GPUs for training')

    # put the model on GPUs
    logger.info(f"Sending the model to '{device}'")
    model = model.to(device)

    # Log the number of learnable parameters
    logger.info(f'Number of learnable params {get_number_of_learnable_parameters(model)}')

    # crate loss
    loss_criterion = _create_loss()
    logger.info(loss_criterion)

    # Create evaluation metric
    eval_criterion = _create_eval()
    logger.info(eval_criterion)

    # Create the optimizer
    optimizer = _create_optimizer(model)
    logger.info(optimizer)

    # create scheduler
    scheduler = _create_scheduler(optimizer)
    logger.info(scheduler)

    # set checkpoint directory
    checkpoint_dir = get_checkpoint_dir()
    logger.info(f"checkpoint directory is {checkpoint_dir}")

    # Create model trainer
    trainer = UNet3DTrainer(model=model, logger=logger, optimizer=optimizer, loss_criterion=loss_criterion,
                            lr_scheduler=scheduler, device=device, eval_criterion=eval_criterion,
                            checkpoint_dir=checkpoint_dir, best_eval_score=cfg.best_eval_score,
                            max_num_epochs=cfg.max_num_epochs,
                            validate_after_iter=cfg.validate_after_iter, log_after_iter=cfg.log_after_iter)

    # Start training
    trainer.train(train_loader, eval_loader)
