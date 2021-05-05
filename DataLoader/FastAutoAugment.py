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
    def __init__(self, model, dataset, optimizer, scheduler, loss_criterion, eval_criterion):
        self.model = model
        self.dataset = dataset
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_criterion = loss_criterion
        self.eval_criterion = eval_criterion
        self.lr_scheduler = scheduler

    def fast_auto_augment(self, K=3, B=100, T=2, N=10, num_process=3):
        num_process = min(torch.cuda.device_count(), num_process)
        transform, futures = [], []

        torch.multiprocessing.set_start_method('spawn', force=True)

        transform_candidates = DEFALUT_CANDIDATES

        # split data set
        # Define the K-fold Cross Validator
        kfold = KFold(n_splits=K, shuffle=True)

        with ProcessPoolExecutor(max_workers=num_process) as executor:
            for fold, (Dm_indx, Da_indx) in enumerate(kfold.split(self.dataset)):
                future = executor.submit(self.process_fn
                                         , Dm_indx, Da_indx, T, transform_candidates, B, N, fold)
                futures.append(future)

            for future in futures:
                transform.extend(future.result())

        transform = transforms.RandomChoice(transform)

        return transform

    def process_fn(self, Dm_indx, Da_indx, T, transform_candidates, B, N, k):
        torch.manual_seed(42)

        # Sample elements randomly from a given list of ids, no replacement.
        train_subsampler = torch.utils.data.SubsetRandomSampler(Dm_indx)
        eval_subsampler = torch.utils.data.SubsetRandomSampler(Da_indx)

        # # Define data loaders for training and testing data in this fold
        # evalloader = torch.utils.data.DataLoader(self.dataset, batch_size=2, sampler=eval_subsampler)

        device_id = k % torch.cuda.device_count()
        device = torch.device('cuda:%d' % device_id)
        _transform = []

        print('[+] Child %d training strated (GPU: %d)' % (k, device_id))

        # train child model
        logger = get_logger(cfg.log_path)
        trainer = self.train_child(train_subsampler, device, logger)

        # search sub policy
        for t in range(T):
            subpolicies = self.search_subpolicies(transform_candidates, eval_subsampler, B, trainer)
            subpolicies = self.get_topn_subpolicies(subpolicies, N)
            _transform.extend([subpolicy[0] for subpolicy in subpolicies])

        return _transform

    def train_child(self, train_subsampler, logger, device):

        self.model.to(device)

        if torch.cuda.device_count() > 1:
            print('\n[+] Use {} GPUs'.format(torch.cuda.device_count()))
            self.model = nn.DataParallel(self.model)

        trainloader = torch.utils.data.DataLoader(self.dataset, batch_size=2, sampler=train_subsampler)

        trainer = UNet3DTrainer(model=self.model, logger=logger, optimizer=self.optimizer,
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

    def search_subpolicies(self, transform_candidates, eval_subsampler, B, trainer):
        def _objective(sampled):
            trainloader = torch.utils.data.DataLoader(self.dataset, batch_size=2, sampler=eval_subsampler)
            subpolicy = [transform(prob, mag) for transform, prob, mag in sampled]


            val_res = trainer.validate(trainloader)

            return {'val_res': val_res, 'status': STATUS_OK}

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
        for t in trials.trials:
            vals = t['misc']['vals']
            subpolicy = [transform_candidates[vals['transform1'][0]](vals['prob1'][0], vals['mag1'][0]),
                         transform_candidates[vals['transform2'][0]](vals['prob2'][0], vals['mag2'][0])]
            subpolicy = transforms.Compose([
                ## baseline augmentation
                transforms.Pad(4),
                transforms.RandomCrop(32),
                transforms.RandomHorizontalFlip(),
                ## policy
                *subpolicy,
                ## to tensor
                transforms.ToTensor()])
            subpolicies.append((subpolicy, t['result']['loss']))

    def get_topn_subpolicies(self, subpolicies, N):
        return sorted(subpolicies, key=lambda subpolicy: subpolicy[1])[:N]

    def reset_weights(self, m):
        '''
          Try resetting model weights to avoid
          weight leakage.
        '''
        for layer in m.children():
            if hasattr(layer, 'reset_parameters'):
                print(f'Reset trainable parameters of layer = {layer}')
                layer.reset_parameters()


