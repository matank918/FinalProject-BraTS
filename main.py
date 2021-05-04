import importlib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from utils.utils import get_number_of_learnable_parameters
from utils.Log import get_logger, get_module_variable
from trainer import UNet3DTrainer
from DataLoader.CustomDataSet import CustomDataset
from Loss.loss import create_loss
from Loss.metrics import create_eval
import utils.config as cfg
from nnUnet.nnUnet3d import UNet3D
from DataLoader.FastAutoAugment import FastAutoAugment
import logging

def _get_model():
    module = importlib.import_module(cfg.module_name)
    basic_block = getattr(module, cfg.basic_block)
    return UNet3D(in_channels=cfg.in_channels, out_channels=cfg.out_channels, f_maps=cfg.f_maps,
                  apply_pooling=cfg.apply_pooling
                  ,basic_module=basic_block, deep_supervision=cfg.deep_supervision)


def _get_train_loaders(dataset):
    n_val = int(len(dataset) * cfg.val_percent)
    n_train = len(dataset) - n_val
    train, val = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train, batch_size=cfg.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val, batch_size=cfg.batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)
    loader_dict = {'train': train_loader, 'val': val_loader}
    return loader_dict


def _get_loss_criterion():
    return create_loss(cfg.loss_name)


def _get_eval_criterion():
    return create_eval(cfg.eval_name)


def _create_trainer(model, device, optimizer, lr_scheduler, loss_criterion, eval_criterion, loaders, logger):
    return UNet3DTrainer(model=model, logger=logger, optimizer=optimizer, loss_criterion=loss_criterion,
                         lr_scheduler=lr_scheduler,
                         eval_criterion=eval_criterion, device=device, loaders=loaders,
                         num_iterations=cfg.num_iterations,
                         validate_iters=cfg.validate_iters, checkpoint_dir=cfg.checkpoint_dir,
                         best_eval_score=cfg.best_eval_score,
                         validate_after_iters=cfg.validate_after_iters,
                         log_after_iters=cfg.log_after_iters,
                         num_epoch=cfg.num_epoch, max_num_iterations=cfg.max_num_iterations,
                         eval_score_higher_is_better=cfg.eval_score_higher_is_better, max_num_epochs=cfg.max_num_epochs,
                         accumulation_steps=cfg.accumulation_steps)


def _create_optimizer(model):
    if cfg.optimizer_name == 'SGD':
        return optim.SGD(model.parameters(), lr=cfg.learning_rate, momentum=cfg.momentum, nesterov=cfg.nesterov)
    elif cfg.optimizer_name == 'Adam':
        return optim.Adam(model.parameters(), lr=cfg.learning_rate)


def _create_lr_scheduler(optimizer):
    lr_lambda = lambda epoch: 0.99 * epoch
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


if __name__ == '__main__':
    # Load and log experiment configuration
    logger = get_logger(cfg.log_path)
    logger.info(cfg.id)
    logger.info(cfg.run_name)
    logger.info(cfg.run_purpose)
    logger.info(get_module_variable(cfg))

    # Create the model
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

    # Create loss criterion
    loss_criterion = _get_loss_criterion()

    # Create evaluation metric
    eval_criterion = _get_eval_criterion()

    # Create data loaders
    dataset = CustomDataset(cfg.loader_path)
    loaders = _get_train_loaders(dataset)

    # Create the optimizer
    optimizer = _create_optimizer(model)

    # Create learning rate adjustment strategy
    lr_scheduler = _create_lr_scheduler(optimizer)

    Auto = FastAutoAugment(model=model, loss_criterion=loss_criterion,optimizer=optimizer, scheduler=lr_scheduler, device=device,
                           eval_criterion=eval_criterion, dataset=dataset)

    # Create model trainer
    # trainer = _create_trainer(model=model, optimizer=optimizer, device=device, logger=logger,
    #                           loss_criterion=loss_criterion,
    #                           eval_criterion=eval_criterion, loaders=loaders,
    #                           lr_scheduler=lr_scheduler)
    # # Start training
    # trainer.fit()
