import importlib

import torch
import torch.nn as nn
from utils import get_logger
import torch.optim as optim
from utils import get_number_of_learnable_parameters
from trainer import UNet3DTrainer
from torch.utils.data import DataLoader, random_split
from BasicDataSet import BasicDataset
import configparser
import pathlib


def _LoadData(config):
    cwd = pathlib.Path.cwd()
    last_dir = cwd.name
    cwd = str(cwd)
    print(cwd)
    last_dir = str(last_dir)
    imgs_dir = cwd.replace(last_dir, config['loader']['path'])
    print(imgs_dir)
    dataset = BasicDataset(imgs_dir)
    val_percent=config['loader']['val percent']
    batch_size=config['loader']['batch size']
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train, val = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)
    loader_dict={'train': train_loader ,'val':val_loader }
    return loader_dict

def _create_trainer(config, model, optimizer, loss_criterion, eval_criterion, loaders):
    assert 'trainer' in config, 'Could not find trainer configuration'
    trainer_config = config['trainer']

    skip_train_validation = trainer_config.get('skip_train_validation', False)

    # get tensorboard formatter
    # tensorboard_formatter = get_tensorboard_formatter(trainer_config.get('tensorboard_formatter', None))

    return UNet3DTrainer(model, optimizer, loss_criterion, eval_criterion,
                         config['device'], loaders, trainer_config['checkpoint_dir'],
                         max_num_iterations=trainer_config['iters'],
                         validate_after_iters=trainer_config['validate_after_iters'],
                         log_after_iters=trainer_config['log_after_iters'],
                         tensorboard_formatter=tensorboard_formatter,
                         skip_train_validation=skip_train_validation)


def _create_optimizer(config, model):
    assert 'optimizer' in config, 'Cannot find optimizer configuration'
    optimizer_config = config['optimizer']
    learning_rate = optimizer_config['learning_rate']
    weight_decay = optimizer_config['weight_decay']
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    return optimizer


def main():
    # Load and log experiment configuration
    config = configparser.ConfigParser()
    config.sections()
    config.read(r"C:\Users\Elinoy\Documents\project\final project\cfg_file.ini")
    print(config['paths']['in_data_path'])
    logger = get_logger('UNet3DTrain')
    logger.info(config)
    LoadData = _LoadData(config)
#
#     manual_seed = config.get('manual_seed', None)
#     if manual_seed is not None:
#         logger.info(f'Seed the RNG for all devices with {manual_seed}')
#         torch.manual_seed(manual_seed)
#         # see https://pytorch.org/docs/stable/notes/randomness.html
#         torch.backends.cudnn.deterministic = True
#         torch.backends.cudnn.benchmark = False
#
#     # Create the model
#     model = get_model(config)
#     # use DataParallel if more than 1 GPU available
#     device = config['device']
#     if torch.cuda.device_count() > 1 and not device.type == 'cpu':
#         model = nn.DataParallel(model)
#         logger.info(f'Using {torch.cuda.device_count()} GPUs for training')
#
#     # put the model on GPUs
#     logger.info(f"Sending the model to '{config['device']}'")
#     model = model.to(device)
#
#     # Log the number of learnable parameters
#     logger.info(f'Number of learnable params {get_number_of_learnable_parameters(model)}')
#
#     # Create loss criterion
#     loss_criterion = get_loss_criterion(config)
#     # Create evaluation metric
#     eval_criterion = get_evaluation_metric(config)
#
#     # Create data loaders
#     loaders = get_train_loaders(config)
#
#     # Create the optimizer
#     optimizer = _create_optimizer(config, model)
#
#     # Create model trainer
#     trainer = _create_trainer(config, model=model, optimizer=optimizer,
#                               loss_criterion=loss_criterion, eval_criterion=eval_criterion, loaders=loaders)
#     # Start training
#     trainer.fit()
#
#
# if __name__ == '__main__':
#     main()
