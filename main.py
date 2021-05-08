import importlib
import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import *
from utils.utils import get_number_of_learnable_parameters
from utils.Log import get_logger, get_module_variable, get_checkpoint_dir
from trainer import UNet3DTrainer
from DataLoader.CustomDataSet import CustomDataset, get_loaders
from Loss.metrics import create_eval
import utils.config as cfg
from Loss.improvedLoss import *
from Loss.loss import *
from nnUnet.nnUnet3d import *

if __name__ == '__main__':
    # Load and log experiment configuration

    logger = get_logger(cfg.log_path)
    logger.info(cfg.id)
    logger.info(cfg.run_name)
    logger.info(cfg.run_purpose)
    logger.info(get_module_variable(cfg))

    # Create the model
    module = importlib.import_module(cfg.module_name)
    basic_block = getattr(module, cfg.basic_block)
    model = UNet3D(in_channels=cfg.in_channels, out_channels=cfg.out_channels, f_maps=cfg.f_maps,
                  apply_pooling=cfg.apply_pooling
                  ,basic_module=basic_block, deep_supervision=cfg.deep_supervision)

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
    loss_criterion = DiceLoss()
    # loss_criterion = SoftDiceLoss()
    # loss_criterion = DC_and_BCE_loss({})

    # Create evaluation metric
    eval_criterion = create_eval(cfg.eval_name)

    # Create data loaders
    dataset = CustomDataset(cfg.loader_path, transforms=(), data_transform=())
    train_loader, eval_loader = get_loaders(dataset, cfg.val_percent, cfg.batch_size)

    # Create the optimizer
    optimizer = Adam(model.parameters(), lr=cfg.initial_lr, weight_decay=cfg.weight_decay, amsgrad=True)
    # optimizer = SGD(model.parameters(), lr=cfg.initial_lr, weight_decay=cfg.weight_decay, nesterov=True, momentum=0.99)

    # Create learning rate adjustment strategy
    lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2,
                      patience=cfg.lr_scheduler_patience,
                      verbose=True, threshold=cfg.lr_scheduler_eps,
                      threshold_mode="abs")

    # lambda1 = lambda epoch: epoch // 30
    # lambda2 = lambda epoch: 0.95 ** epoch
    # scheduler = LambdaLR(optimizer, lr_lambda=[lambda1, lambda2])

    # set checkpoint directory
    checkpoint_dir = get_checkpoint_dir()

    # Create model trainer
    trainer = UNet3DTrainer(model=model, logger=logger, optimizer=optimizer, loss_criterion=loss_criterion,
                            lr_scheduler=None, device=device, eval_criterion=eval_criterion,
                            checkpoint_dir=checkpoint_dir, best_eval_score=cfg.best_eval_score,
                            max_num_epochs=cfg.max_num_epochs, accumulation_steps=cfg.accumulation_steps,
                            validate_after_iter=cfg.validate_after_iter, log_after_iter=cfg.log_after_iter)
    # Start training
    trainer.train(train_loader, eval_loader)

