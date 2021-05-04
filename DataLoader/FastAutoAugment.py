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
import torch
from DataLoader.CustomTransformations import *
from utils.utils import split_dataset


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
    def __init__(self, model, dataset, device, optimizer, scheduler, loss_criterion, eval_criterion):
        self.model = model
        self.dataset = dataset
        self.device = device
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_criterion = loss_criterion
        self.eval_criterion = eval_criterion


    def fast_auto_augment(self, K=5, B=100, T=2, N=10, num_process=5):
        num_process = min(torch.cuda.device_count(), num_process)
        transform, futures = [], []

        torch.multiprocessing.set_start_method('spawn', force=True)

        transform_candidates = DEFALUT_CANDIDATES

        # split data set
        Dm_indexes, Da_indexes = split_dataset(self.trainloader, K)

        with ProcessPoolExecutor(max_workers=num_process) as executor:
            for k, (Dm_indx, Da_indx) in enumerate(zip(Dm_indexes, Da_indexes)):
                future = executor.submit(self.process_fn
                                        ,Dm_indx, Da_indx, T, transform_candidates, B, N, k)
                futures.append(future)

            for future in futures:
                transform.extend(future.result())

        transform = transforms.RandomChoice(transform)

        return transform

    def process_fn(self, Dm_indx, Da_indx, T, transform_candidates, B, N, k):
        device_id = k % torch.cuda.device_count()
        device = torch.device('cuda:%d' % device_id)
        _transform = []

        print('[+] Child %d training strated (GPU: %d)' % (k, device_id))

        # train child model
        child_model = copy.deepcopy(self.model)
        logger = get_logger(cfg.log_path)
        self.train_child(child_model, self.trainloader, Dm_indx, device, logger)

        # # search sub policy
        # for t in range(T):
        #     subpolicies = self.search_subpolicies_hyperopt(transform_candidates, child_model, self.trainloader, Da_indx, B, device)
        #     subpolicies = self.get_topn_subpolicies(subpolicies, N)
        #     _transform.extend([subpolicy[0] for subpolicy in subpolicies])
        #
        # return _transform

    def train_child(self, model, dataset, subset_indx, logger, device=None):

        dataset.transform = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor()])
        subset = Subset(dataset, subset_indx)

        if device:
            model = model.to(device)

            if torch.cuda.device_count() > 1:
                print('\n[+] Use {} GPUs'.format(torch.cuda.device_count()))
                self.model = nn.DataParallel(self.model)

        for step in range(len(subset)):
            batch = next(self.trainloader)
            trainer = trainerCNN(model=model, criterion=self.criterion, epochs=cfg.epochs, optimizer=self.optimizer,
                                 device=device,
                                 log_iter=cfg.log_iter, trainloader=batch, evalloader=None, logger=logger,
                                 evel_iter=cfg.eval_iter, checkpoint_dir=None,
                                 best_eval_score=cfg.best_eval_score)
            trainer.fit()

    def validate_child(self, model, dataset, subset_indx, transform, device=None):
        criterion = nn.CrossEntropyLoss()

        if device:
            model = model.to(device)

        self.trainloader.transform = transform
        subset = Subset(dataset, subset_indx)
        # data_loader = get_dataloader(args, subset, pin_memory=False)
        #
        # return validate(args, model, criterion, data_loader, 0, None, device)


    def search_subpolicies_hyperopt(self, transform_candidates, child_model, Da_indx, B):
        def _objective(sampled):
            subpolicy = [transform(prob, mag)
                         for transform, prob, mag in sampled]

            subpolicy = transforms.Compose([
                transforms.Resize(32),
                *subpolicy,
                transforms.ToTensor()])

            val_res = self.validate_child(child_model, Da_indx, subpolicy, self.device)
            loss = val_res[2].cpu().numpy()
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


if __name__ == '__main__':
    model = CifarCNN()
    logger = get_logger(cfg.log_path)
    trainloader, _, _ = get_data(cfg.svhn_path, cfg.batch_size, 0, 0)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    Fast = FastAutoAugment(model=model, trainloader=trainloader, optimizer=optimizer,
                           criterion=criterion, scheduler=None, device=device)
    Fast.fast_auto_augment()