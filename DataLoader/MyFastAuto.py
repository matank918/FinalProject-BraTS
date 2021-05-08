import copy
import json
import time
import torch
import random
import torchvision.transforms as transforms
from torch.utils.data import Subset
from concurrent.futures import ProcessPoolExecutor
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import utils.config as cfg
from sklearn.model_selection import KFold, StratifiedShuffleSplit
import torch.optim as optim
from utils.Log import get_logger, get_module_variable
from utils.utils import *

from trainer import UNet3DTrainer
from DataLoader.CustomDataSet import *
from Loss.loss import create_loss
from Loss.metrics import create_eval
from nnUnet.nnUnet3d import get_model
from pathlib import Path
import torchio as tio
from DataLoader.BasicTransformations import *
from DataLoader.GeometricTransformations import *

GeometricTransform = [Rotate]


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

    def fast_auto_augment(self, K=5, B=20, T=1, N=10, num_process=3):
        """

        :param K: number of folds for the k-cross validation
        :param B: number of epoch for the hyperopt search
        :param T: number of Searches for each epoch
        :param N: number of subpolicies
        :param num_process:
        :return:
        """

        num_process = min(torch.cuda.device_count(), num_process)
        transform, futures = [], []

        torch.multiprocessing.set_start_method('spawn', force=True)

        transform_candidates = GeometricTransform

        # split data set
        # Define the K-fold Cross Validator

        kfold = KFold(n_splits=K, shuffle=True)

        for fold, (Dm_indx, Da_indx) in enumerate(kfold.split(self.dataset)):
            transform = self.process_fn(Dm_indx, Da_indx, T, transform_candidates, B, N, fold)
            transform.extend(transform)

        return transform

    def process_fn(self, Dm_indx, Da_indx, T, transform_candidates, B, N, fold):

        device_id = fold % torch.cuda.device_count()
        device = torch.device('cuda:%d' % device_id)
        _transform = []
        print('[+] Child %d training strated (GPU: %d)' % (fold, device_id))

        self.reset_weights(model)
        # start training
        trainer = self.train(Dm_indx, device=device)

        for t in range(T):
            subpolicies = self.search_subpolicies(transform_candidates, Da_indx, trainer, B)
            subpolicies = self.get_topn_subpolicies(subpolicies, N)
            _transform.extend([subpolicy[0] for subpolicy in subpolicies])
        return _transform

    def train(self, Dm_indx, device):
        train_subset = Subset(self.dataset, Dm_indx)
        trainloader = torch.utils.data.DataLoader(train_subset, batch_size=cfg.batch_size)
        self.model.to(device)

        trainer = UNet3DTrainer(model=self.model, logger=self.logger, optimizer=self.optimizer,
                                loss_criterion=self.loss_criterion,
                                lr_scheduler=self.lr_scheduler,
                                eval_criterion=self.eval_criterion, device=device,
                                best_eval_score=cfg.best_eval_score,
                                log_after_iters=cfg.log_after_iters,
                                max_num_epochs=1,
                                accumulation_steps=cfg.accumulation_steps)

        trainer.train(trainloader, None)

        return trainer

    def search_subpolicies(self, transform_candidates, Da_indx, trainer, B):

        def _objective(sampled):
            subpolicy = [transform(prob, mag)
                         for transform, prob, mag in sampled]

            subpolicy = Compose([
            ToTensor(),
            RandomCrop3D((240, 240, 155), (128, 128, 128)),
            # tio.transforms.RandomFlip(flip_probability=0.2),
            *subpolicy
        ])
            loss, _ = self.validate(Da_indx, trainer, subpolicy)
            return {'loss': loss, 'status': STATUS_OK}

        space = [
            (hp.choice('transform1', transform_candidates), hp.uniform('prob1', 0, 1.0), hp.uniform('mag1', 0, 1.0)),
            (hp.choice('transform2', transform_candidates), hp.uniform('prob2', 0, 1.0), hp.uniform('mag2', 0, 1.0))]

        trials = Trials()
        best = fmin(_objective,
                    space=space,
                    algo=tpe.suggest,
                    max_evals=B,
                    trials=trials)

        subpolicies = []
        print(len(trials.trials))
        for t in trials.trials:
            vals = t['misc']['vals']
            subpolicy = [transform_candidates[vals['transform1'][0]](vals['prob1'][0], vals['mag1'][0]),
                         transform_candidates[vals['transform2'][0]](vals['prob2'][0], vals['mag2'][0])]
            subpolicy = transforms.Compose([
                ## to tensor
                ToTensor(),
                ## policy
                *subpolicy])
            subpolicies.append((subpolicy, t['result']['loss']))

        print(subpolicies)

        return subpolicies

    def validate(self, Da_indx, trainer, transform):
        self.dataset.transform = transform
        eval_subset = Subset(self.dataset, Da_indx)
        evalloader = torch.utils.data.DataLoader(eval_subset, batch_size=cfg.batch_size)
        evel_result = trainer.validate(evalloader)

        return evel_result

    @staticmethod
    def get_topn_subpolicies(subpolicies, N):
        return sorted(subpolicies, key=lambda subpolicy: subpolicy[1])[:N]

    @staticmethod
    def reset_weights(m):
        '''
          Try resetting model weights to avoid
          weight leakage.
        '''
        for layer in m.children():
            if hasattr(layer, 'reset_parameters'):
                print(f'Reset trainable parameters of layer = {layer}')
                layer.reset_parameters()


if __name__ == '__main__':
    one_up = Path(__file__).parents[1]
    os.chdir(one_up)

    logger = get_logger(cfg.log_path)

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
    # train_loader, eval_loader = get_loaders(dataset, cfg.val_percent, cfg.batch_size)

    # Create the optimizer
    optimizer = optim.SGD(model.parameters(), lr=cfg.learning_rate, momentum=cfg.momentum, nesterov=cfg.nesterov)

    # Create learning rate adjustment strategy
    lr_lambda = lambda epoch: 0.99 * epoch
    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    Auto = FastAutoAugment(model=model, loss_criterion=loss_criterion, optimizer=optimizer,
                           scheduler=lr_scheduler,
                           eval_criterion=eval_criterion, dataset=dataset, logger=logger)

    print(Auto.fast_auto_augment())
