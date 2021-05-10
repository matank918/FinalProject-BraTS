import importlib
import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import *
from utils.utils import get_number_of_learnable_parameters
from utils.Log import get_logger, get_module_variable, get_checkpoint_dir
from trainer import UNet3DTrainer
from DataLoader.CustomDataSet import CustomDataset, get_loaders
# from Loss.metrics import create_eval
import utils.config as cfg
# from Loss.loss import *
from nnUnet.nnUnet3d import *
from monai.losses import DiceLoss, DiceCELoss
from monai.metrics import DiceMetric
from DataLoader.BasicTransformations import train_transforms
from monai.networks.layers.factories import Act, Norm
import os


def _create_loss():
    # Create loss criterion
    if cfg.loss_name == "Dice":
        return DiceLoss(to_onehot_y=False, squared_pred=True, softmax=True, include_background=cfg.include_background)
    if cfg.loss_name == "DiceCE":
        return DiceCELoss(to_onehot_y=False, squared_pred=True, softmax=True, include_background=cfg.include_background)


def _create_optimizer(model):
    # Create loss criterion
    if cfg.optimizer_name == "Adam":
        return Adam(model.parameters(), lr=cfg.initial_lr, weight_decay=cfg.weight_decay, amsgrad=cfg.amsgrad)
    if cfg.optimizer_name == "SGD":
        return SGD(model.parameters(), lr=cfg.initial_lr, nesterov=cfg.nesterov, momentum=cfg.momentum)


def _create_scheduler(optimizer):
    lambda1 = lambda epoch: 0.999 ** epoch
    return LambdaLR(optimizer, lr_lambda=lambda1)


def _create_eval():
    if cfg.eval_name == "DiceMetric":
        return DiceMetric(include_background=cfg.include_background)


if __name__ == '__main__':
    # Load and log experiment configuration
    logger = get_logger(cfg.log_path)
    logger.info(cfg.id)
    logger.info(cfg.run_name)
    logger.info(cfg.run_purpose)
    logger.info(get_module_variable(cfg))

    # Create data loaders
    dataset = CustomDataset(cfg.loader_path, transform=train_transforms)
    train_loader, eval_loader = get_loaders(dataset, cfg.val_percent, cfg.batch_size)

    model = UNet(
        dimensions=3,
        in_channels=4,
        out_channels=4,
        channels=(32, 64, 128, 256, 320, 320),
        strides=(2, 2, 2, 2, 2),
        num_res_units=0,
        act=Act.LEAKYRELU,
        norm=Norm.BATCH)

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
                            max_num_epochs=cfg.max_num_epochs, accumulation_steps=cfg.accumulation_steps,
                            validate_after_iter=cfg.validate_after_iter, log_after_iter=cfg.log_after_iter)

    # Start training
    trainer.train(train_loader, eval_loader)


