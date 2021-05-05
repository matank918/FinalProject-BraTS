import copy
import json
import time
import torch
import random
import torchvision.transforms as transforms
from torch.utils.data import Subset
from sklearn.model_selection import StratifiedShuffleSplit
from concurrent.futures import ProcessPoolExecutor
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from utils.Log import get_logger, get_module_variable, get_log_name
import utils.config as cfg
import torch
from DataLoader.CustomTransformations import *
from utils.utils import split_dataset
from trainer import UNet3DTrainer
from DataLoader.CustomDataSet import get_loaders
from sklearn.model_selection import KFold
import importlib
import torch
import torch.nn as nn
import torch.optim as optim
from utils.utils import get_number_of_learnable_parameters
from utils.Log import get_logger, get_module_variable
from trainer import UNet3DTrainer
from DataLoader.CustomDataSet import CustomDataset, get_loaders
from Loss.loss import create_loss
from Loss.metrics import create_eval
from nnUnet.nnUnet3d import UNet3D
from nnUnet.nnUnet3d import get_model

DEFALUT_CANDIDATES = [
    ShearXY,
    TranslateXY,
    Rotate,
    AutoContrast,
    Invert,
    Equalize,
    Solarize,
    Posterize,
    Contrast,
    Color,
    Brightness,
    Sharpness,
    Cutout,
    #     SamplePairing,
]


class FastAutoAugment:
    def __init__(self, model, dataset, optimizer, scheduler, loss_criterion, eval_criterion, logger):
        self.model = model
        self.dataset = dataset
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_criterion = loss_criterion
        self.eval_criterion = eval_criterion
        self.lr_scheduler = scheduler
        self.logger = logger

    def fast_auto_augment(self, K=3, B=100, T=2, N=10, num_process=3):
        num_process = min(torch.cuda.device_count(), num_process)
        transform, futures = [], []

        torch.multiprocessing.set_start_method('spawn', force=True)

        transform_candidates = DEFALUT_CANDIDATES

        # Set fixed random number seed
        torch.manual_seed(42)

        # split data set
        # Define the K-fold Cross Validator
        kfold = KFold(n_splits=K, shuffle=True)

        for fold, (Dm_indx, Da_indx) in enumerate(kfold.split(self.dataset)):
            transform = self.process_fn(Dm_indx, Da_indx, T, transform_candidates, B, N, fold)
            transform.extend(transform)

        transform = transforms.RandomChoice(transform)

        return transform

    def process_fn(self, Dm_indx, Da_indx, T, transform_candidates, B, N, fold):

        device_id = fold % torch.cuda.device_count()
        device = torch.device('cuda:%d' % device_id)
        _transform = []
        print('[+] Child %d training strated (GPU: %d)' % (fold, device_id))

        self.reset_weights()
        # start training
        trainer = self.train(Dm_indx, device=device)

        for t in range(T):
            subpolicies = self.search_subpolicies(trainer)
            subpolicies = self.get_topn_subpolicies(subpolicies, N)
            _transform.extend([subpolicy[0] for subpolicy in subpolicies])
        return _transform

    def train(self, Dm_indx, device):
        trainer = UNet3DTrainer(model=self.model, logger=self.logger, optimizer=self.optimizer,
                                loss_criterion=self.loss_criterion,
                                lr_scheduler=self.lr_scheduler,
                                eval_criterion=self.eval_criterion, device=device,
                                checkpoint_dir=cfg.checkpoint_dir,
                                best_eval_score=cfg.best_eval_score,
                                validate_after_iters=cfg.validate_after_iters,
                                log_after_iters=cfg.log_after_iters,
                                max_num_iterations=cfg.max_num_iterations,
                                max_num_epochs=cfg.max_num_epochs,
                                accumulation_steps=cfg.accumulation_steps)
        trainer.fit(trainloader, None)

        return trainer

    def search_subpolicies(self, Da_indx):
        print("serching")
        return []

    def validate(self, Da_indx):
        pass

    def get_topn_subpolicies(self, subpolicies, N):
        return sorted(subpolicies, key=lambda subpolicy: subpolicy[1])[:N]

    def reset_weights(self):
        '''
          Try resetting model weights to avoid
          weight leakage.
        '''
        for layer in self.model.children():
            if hasattr(layer, 'reset_parameters'):
                print(f'Reset trainable parameters of layer = {layer}')
                layer.reset_parameters()


if __name__ == '__main__':
    logger = get_logger(cfg.eval_name)

    # Create the model
    model = get_model(in_channels=cfg.in_channels, out_channels=cfg.out_channels, f_maps=cfg.f_maps,
                      apply_pooling=cfg.apply_pooling, deep_supervision=cfg.deep_supervision,
                      module_name=cfg.module_name,
                      basic_block=cfg.basic_block)

    # Create loss criterion
    loss_criterion = create_loss(cfg.loss_name)

    # Create evaluation metric
    eval_criterion = create_eval(cfg.eval_name)

    # Create data loaders
    dataset = CustomDataset(cfg.loader_path)
    train_loader, eval_loader = get_loaders(dataset, cfg.val_percent, cfg.batch_size)

    # Create the optimizer
    optimizer = optim.SGD(model.parameters(), lr=cfg.learning_rate, momentum=cfg.momentum, nesterov=cfg.nesterov)

    # Create learning rate adjustment strategy
    lr_lambda = lambda epoch: 0.99 * epoch
    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    Auto = FastAutoAugment(model=model, loss_criterion=loss_criterion, optimizer=optimizer,
                           scheduler=lr_scheduler,
                           eval_criterion=eval_criterion, dataset=dataset, logger=logger)

    Auto.fast_auto_augment()
