import importlib

import torch
import torch.nn as nn
from utils import get_logger
import torch.optim as optim
from utils import get_number_of_learnable_parameters, correct_type
from trainer import UNet3DTrainer
from torch.utils.data import DataLoader, random_split
from BasicDataSet import BasicDataset
import configparser
from loss import create_loss


def get_model(config):
    def _model_class(class_name):
        m = importlib.import_module('nnUnet.nnUnet3d')
        clazz = getattr(m, class_name)
        return clazz

    model_config = correct_type(dict(config['model'].items()))
    model_class = _model_class(model_config.pop('name'))
    return model_class(**model_config)

def get_train_loaders(config):
    loader_config = correct_type(dict(config['loader'].items()))
    img_dir = loader_config['path']
    batch_size= int(config['loader']['batch size'])
    val_percent = float(config['loader']['val_percent'])
    dataset = BasicDataset(img_dir)
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train, val = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)
    loader_dict = {'train': train_loader, 'val': val_loader}
    return loader_dict


def get_loss_criterion(config):
    """
    Returns the loss function based on provided configuration
    :param config: (dict) a top level configuration object containing the 'loss' key
    :return: an instance of the loss function
    """
    assert 'loss' in config, 'Could not find loss function configuration'
    loss_config = config['loss']
    name = loss_config.pop('name')

    loss = create_loss(name, loss_config)

    return loss


def _create_trainer(config, model, device, optimizer, loss_criterion, eval_criterion, loaders, logger):
    assert 'train' in config, 'Could not find train configuration'
    train_config = correct_type(dict(config['train'].items()))

    return UNet3DTrainer(model=model, optimizer=optimizer, loss_criterion=loss_criterion,
                         eval_criterion=eval_criterion, logger=logger,
                         device=device, loaders=loaders, validate_iters=train_config['validate_iters'],
                         skip_train_validation = train_config['skip_train_validation'],
                         num_epoch=train_config['iters'],
                         validate_after_iters=train_config['validate_after_iters'],
                         log_after_iters=train_config['log_after_iters'])


def _create_optimizer(config, model):
    assert 'optimizer' in config, 'Cannot find optimizer configuration'
    optimizer_config = correct_type(config['optimizer'])
    learning_rate = optimizer_config['learning_rate']
    weight_decay = optimizer_config['weight_decay']
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    return optimizer


if __name__ == '__main__':
    # Load and log experiment configuration
    config = configparser.ConfigParser()
    config.sections()
    config.read("cfg_file.ini")
    logger = get_logger('UNet3DTrain')
    logger.info(config)

    # Create the model
    model = get_model(config)

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
    loss_criterion = get_loss_criterion(config)

    # Create evaluation metric
    eval_criterion = loss_criterion

    # Create data loaders
    loaders = get_train_loaders(config)

    # Create the optimizer
    optimizer = _create_optimizer(config, model)
    # Create model trainer
    trainer = _create_trainer(config, model=model, optimizer=optimizer, device=device, logger=logger,
                              loss_criterion=loss_criterion, eval_criterion=eval_criterion, loaders=loaders)
    # Start training
    trainer.fit()


