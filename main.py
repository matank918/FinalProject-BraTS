import importlib
import torch
import torch.nn as nn
import torch.optim as optim
from utils.utils import get_number_of_learnable_parameters
from utils.Log import get_logger, get_module_variable, get_checkpoint_dir
from trainer import UNet3DTrainer
from DataLoader.CustomDataSet import CustomDataset, get_loaders
from Loss.loss import create_loss
from Loss.metrics import create_eval
import utils.config as cfg
from nnUnet.nnUnet3d import get_model
import logging


def _create_optimizer(model):
    if cfg.optimizer_name == 'SGD':
        return optim.SGD(model.parameters(), lr=cfg.learning_rate, momentum=cfg.momentum, nesterov=cfg.nesterov)
    elif cfg.optimizer_name == 'Adam':
        return optim.Adam(model.parameters(), lr=cfg.learning_rate)


if __name__ == '__main__':
    # Load and log experiment configuration

    logger = get_logger(cfg.log_path)
    logger.info(cfg.id)
    logger.info(cfg.run_name)
    logger.info(cfg.run_purpose)
    logger.info(get_module_variable(cfg))

    # Create the model
    model = get_model(in_channels=cfg.in_channels, out_channels=cfg.out_channels, f_maps=cfg.f_maps,
                       apply_pooling=cfg.apply_pooling, deep_supervision=cfg.deep_supervision, module_name=cfg.module_name,
                       basic_block=cfg.basic_block)

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
    loss_criterion = create_loss(cfg.loss_name)

    # Create evaluation metric
    eval_criterion = create_eval(cfg.eval_name)

    # Create data loaders
    dataset = CustomDataset(cfg.loader_path, transforms=(), data_transform=())
    train_loader, eval_loader = get_loaders(dataset, cfg.val_percent, cfg.batch_size)

    # Create the optimizer
    optimizer = _create_optimizer(model)

    # Create learning rate adjustment strategy
    lr_lambda = lambda epoch: 0.99 * epoch
    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    checkpoint_dir = get_checkpoint_dir()

    # Create model trainer
    trainer = UNet3DTrainer(model=model, logger=logger, optimizer=optimizer, loss_criterion=loss_criterion,
                            lr_scheduler=lr_scheduler, device=device,
                            eval_criterion=eval_criterion,
                            checkpoint_dir=checkpoint_dir,
                            best_eval_score=cfg.best_eval_score,
                            validate_after_iters=cfg.validate_after_iters,
                            log_after_iters=cfg.log_after_iters,
                            max_num_epochs=cfg.max_num_epochs,
                            accumulation_steps=cfg.accumulation_steps)
    # Start training
    trainer.train(train_loader, eval_loader)

    # Auto = FastAutoAugment(model=model, loss_criterion=loss_criterion,optimizer=optimizer,
    #                        scheduler=lr_scheduler, device=device,
    #                        eval_criterion=eval_criterion, dataset=dataset)
