import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from utils import RunningAverage, save_checkpoint, split_image, split_channels
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time


class UNet3DTrainer:
    """3D UNet trainer.
      Args:
        model (Unet3D): UNet 3D model to be trained
        optimizer (nn.optim.Optimizer): optimizer used for training
        lr_scheduler (torch.optim.lr_scheduler._LRScheduler): learning rate scheduler
            WARN: bear in mind that lr_scheduler.step() is invoked after every validation step
            (i.e. validate_after_iters) not after every epoch. So e.g. if one uses StepLR with step_size=30
            the learning rate will be adjusted after every 30 * validate_after_iters iterations.
        loss_criterion (callable): loss function
        eval_criterion (callable): used to compute training/validation metric (such as Dice, IoU, AP or Rand score)
            saving the best checkpoint is based on the result of this function on the validation set
        device (torch.device): device to train on
        loaders (dict): 'train' and 'val' loaders
        checkpoint_dir (string): dir for saving checkpoints and tensorboard logs
        max_num_epochs (int): maximum number of epochs
        max_num_iterations (int): maximum number of iterations
        validate_after_iters (int): validate after that many iterations
        log_after_iters (int): number of iterations before logging to tensorboard
        validate_iters (int): number of validation iterations, if None validate
            on the whole validation set
        eval_score_higher_is_better (bool): if True higher eval scores are considered better
        best_eval_score (float): best validation score so far (higher better)
        num_iterations (int): useful when loading the model from the checkpoint
        num_epoch (int): useful when loading the model from the checkpoint
    """

    def __init__(self, model, logger, optimizer, loss_criterion, lr_scheduler,
                 eval_criterion, device, loaders, num_iterations
                 ,validate_iters, checkpoint_dir, best_eval_score,
                 validate_after_iters, log_after_iters, num_epoch, max_num_iterations, eval_score_higher_is_better,
                 max_num_epochs):

        self.logger = logger
        self.writer = SummaryWriter()

        self.model = model
        self.optimizer = optimizer
        self.scheduler = lr_scheduler
        self.loss_criterion = loss_criterion
        self.eval_criterion = eval_criterion
        self.device = device
        self.loaders = loaders
        self.checkpoint_dir = checkpoint_dir
        self.max_num_epochs = max_num_epochs
        self.max_num_iterations = max_num_iterations
        self.validate_after_iters = validate_after_iters
        self.log_after_iters = log_after_iters
        self.validate_iters = validate_iters
        self.eval_score_higher_is_better = eval_score_higher_is_better

        self.num_iterations = num_iterations
        self.num_epoch = num_epoch

        self.logger.info(model)
        logger.info(f'eval_score_higher_is_better: {eval_score_higher_is_better}')

        if best_eval_score is not None:
            self.best_eval_score = best_eval_score
        else:
            # initialize the best_eval_score
            if eval_score_higher_is_better:
                self.best_eval_score = float('-inf')
            else:
                self.best_eval_score = float('+inf')

    def fit(self):

        for _ in range(self.num_epoch, self.max_num_epochs):
            # train for one epoch
            should_terminate = self.train()

            if should_terminate:
                self.logger.info('Stopping criterion is satisfied. Finishing training')
                return

            self.num_epoch += 1

        self.logger.info(f"Reached maximum number of epochs: {self.max_num_epochs}. Finishing training...")

    def train(self):
        """Trains the model for 1 epoch.
        Returns:
            True if the training should be terminated immediately, False otherwise
        """
        train_losses = RunningAverage()
        train_eval_scores = RunningAverage()

        # sets the model in training mode
        self.model.train()
        for batch in self.loaders['train']:

            self.logger.info(f'Training iteration [{self.num_iterations}/{self.max_num_iterations}]. '
                             f'Epoch [{self.num_epoch}/{self.max_num_epochs}]')

            # splits between input and target
            input, target = self._split_training_batch(batch)

            # get output from the net, calculate the loss
            output, loss = self._forward_pass(input, target)

            train_losses.update(loss.item(), self._batch_size(input))

            # compute gradients and update parameters
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if self.num_iterations % self.validate_after_iters == 0:
                # set the model in eval mode
                self.model.eval()
                # evaluate on validation set
                eval_score = self.validate()
                # set the model back to training mode
                self.model.train()

                # adjust learning rate if necessary
                if self.scheduler is not None:
                    if isinstance(self.scheduler, ReduceLROnPlateau):
                        self.scheduler.step(eval_score)
                    else:
                        self.scheduler.step()

                # log current learning rate in tensorboard
                self._log_lr()

                # remember best validation metric
                is_best = self._is_best_eval_score(eval_score)

                # save checkpoint
                self._save_checkpoint(is_best)

            if self.num_iterations % self.log_after_iters == 0:

                # compute eval criterion
                eval_score = self.eval_criterion(output, target)
                train_eval_scores.update(eval_score.item(), self._batch_size(input))

                # log stats, params and images
                self.logger.info(
                    f'Training stats. Loss: {train_losses.avg}. Evaluation score: {train_eval_scores.avg}')

                self._log_stats('train', train_losses.avg, train_eval_scores.avg)
                self._log_params()
                # self._log_images(input, target, output, 'train_')

                train_losses = RunningAverage()
                train_eval_scores = RunningAverage()

            if self.should_stop():
                return True

            self.num_iterations += 1

        return False

    def validate(self):

        self.logger.info('Validating...')

        val_losses = RunningAverage()
        val_scores = RunningAverage()

        with torch.no_grad():
            for i, batch in enumerate(self.loaders['val']):

                input, target = self._split_training_batch(batch)

                output, loss = self._forward_pass(input, target)
                val_losses.update(loss.item(), self._batch_size(input))

                if i % 100 == 0:
                    self._log_images(input, target, output, 'val_')

                eval_score = self.eval_criterion(output, target)
                val_scores.update(eval_score.item(), self._batch_size(input))

                if i >= self.validate_iters:
                    # stop validation
                    break

            self._log_stats('val', val_losses.avg, val_scores.avg)
            self.logger.info(f'Validation finished. Loss: {val_losses.avg}. Evaluation score: {val_scores.avg}')
            return val_scores.avg

    def _split_training_batch(self, batch):
        input, target = batch
        input = input.to(self.device)
        target = target.to(self.device)

        return input, target

    def _forward_pass(self, input, target):
        # forward pass
        output = self.model(input)

        # compute the loss
        loss = self.loss_criterion(output, target)

        return output, loss

    def should_stop(self):
        """
        Training will terminate if maximum number of iterations is exceeded or the learning rate drops below
        some predefined threshold (1e-6 in our case)
        """

        if self.max_num_iterations < self.num_iterations:
            self.logger.info(f'Maximum number of iterations {self.max_num_iterations} exceeded.')
            return True

        min_lr = 1e-6
        lr = self.optimizer.param_groups[0]['lr']
        if lr < min_lr:
            self.logger.info(f'Learning rate below the minimum {min_lr}.')
            return True

        return False

    def _is_best_eval_score(self, eval_score):
        if self.eval_score_higher_is_better:
            is_best = eval_score > self.best_eval_score
        else:
            is_best = eval_score < self.best_eval_score

        if is_best:
            self.logger.info(f'Saving new best evaluation metric: {eval_score}')
            self.best_eval_score = eval_score

        return is_best

    def _save_checkpoint(self, is_best):
        # remove `module` prefix from layer names when using `nn.DataParallel`
        # see: https://discuss.pytorch.org/t/solved-keyerror-unexpected-key-module-encoder-embedding-weight-in-state-dict/1686/20
        if isinstance(self.model, nn.DataParallel):
            state_dict = self.model.module.state_dict()
        else:
            state_dict = self.model.state_dict()

        save_checkpoint({
            'epoch': self.num_epoch + 1,
            'num_iterations': self.num_iterations,
            'model_state_dict': state_dict,
            'best_eval_score': self.best_eval_score,
            'eval_score_higher_is_better': self.eval_score_higher_is_better,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'device': str(self.device),
            'max_num_epochs': self.max_num_epochs,
            'max_num_iterations': self.max_num_iterations,
            'validate_after_iters': self.validate_after_iters,
            'log_after_iters': self.log_after_iters,
            'validate_iters': self.validate_iters,
        }, is_best, checkpoint_dir=self.checkpoint_dir,
            logger=self.logger)

    def _log_lr(self):
        lr = self.optimizer.param_groups[0]['lr']
        self.writer.add_scalar('learning_rate', lr, self.num_iterations)

    def _log_stats(self, phase, loss_avg, eval_score_avg):
        tag_value = {
            f'{phase}_loss_avg': loss_avg,
            f'{phase}_eval_score_avg': eval_score_avg
        }

        for tag, value in tag_value.items():
            self.writer.add_scalar(tag, value, self.num_iterations)

    def _log_params(self):
        for name, value in self.model.named_parameters():
            self.writer.add_histogram(name, value.data.cpu().numpy(), self.num_iterations)
            self.writer.add_histogram(name + '/grad', value.grad.data.cpu().numpy(), self.num_iterations)

    def _log_images(self, input, target, prediction, prefix):

        _, ch1, ch2, ch4 = split_channels(target)
        Sagittal_ch1, Coronal_ch1, Horizontal_ch1 = split_image(ch1)
        Sagittal_ch2, Coronal_ch2, Horizontal_ch2 = split_image(ch2)
        Sagittal_ch4, Coronal_ch4, Horizontal_ch4 = split_image(ch4)

        _, pred_ch1, pred_ch2, pred_ch4 = split_channels(prediction)
        Sagittal_pred_ch1, Coronal_pred_ch1, Horizontal_pred_ch1 = split_image(pred_ch1)
        Sagittal_pred_ch2, Coronal_pred_ch2, Horizontal_pred_ch2 = split_image(pred_ch2)
        Sagittal_pred_ch4, Coronal_pred_ch4, Horizontal_pred_ch4 = split_image(pred_ch2)

        self.writer.add_image(prefix + "seg_Sagittal_ch1", Sagittal_ch1, self.num_iterations, dataformats='CHW')
        self.writer.add_image(prefix + "seg_Coronal_ch1", Coronal_ch1, self.num_iterations, dataformats='CHW')
        self.writer.add_image(prefix + "seg_Horizontal_ch1", Horizontal_ch1, self.num_iterations, dataformats='CHW')
        self.writer.add_image(prefix + "seg_Sagittal_ch2", Sagittal_ch2, self.num_iterations, dataformats='CHW')
        self.writer.add_image(prefix + "seg_Coronal_ch2", Coronal_ch2, self.num_iterations, dataformats='CHW')
        self.writer.add_image(prefix + "seg_Horizontal_ch2", Horizontal_ch2, self.num_iterations, dataformats='CHW')
        self.writer.add_image(prefix + "seg_Sagittal_ch4", Sagittal_ch4, self.num_iterations, dataformats='CHW')
        self.writer.add_image(prefix + "seg_Coronal_ch4", Coronal_ch4, self.num_iterations, dataformats='CHW')
        self.writer.add_image(prefix + "seg_Horizontal_ch4", Horizontal_ch4, self.num_iterations, dataformats='CHW')
        self.writer.add_image(prefix + "pred_Sagittal_ch1", Sagittal_pred_ch1, self.num_iterations, dataformats='CHW')
        self.writer.add_image(prefix + "pred_Coronal_ch1", Coronal_pred_ch1, self.num_iterations, dataformats='CHW')
        self.writer.add_image(prefix + "pred_Horizontal_ch1", Horizontal_pred_ch1, self.num_iterations, dataformats='CHW')
        self.writer.add_image(prefix + "pred_Sagittal_ch2", Sagittal_pred_ch2, self.num_iterations, dataformats='CHW')
        self.writer.add_image(prefix + "pred_Coronal_ch2", Coronal_pred_ch2, self.num_iterations, dataformats='CHW')
        self.writer.add_image(prefix + "pred_Horizontal_ch2", Horizontal_pred_ch2, self.num_iterations, dataformats='CHW')
        self.writer.add_image(prefix + "pred_Sagittal_ch4", Sagittal_pred_ch4, self.num_iterations, dataformats='CHW')
        self.writer.add_image(prefix + "pred_Coronal_ch4", Coronal_pred_ch4, self.num_iterations, dataformats='CHW')
        self.writer.add_image(prefix + "pred_Horizontal_ch4", Horizontal_pred_ch4, self.num_iterations, dataformats='CHW')

    @staticmethod
    def _batch_size(input):
        if isinstance(input, list) or isinstance(input, tuple):
            return input[0].size(0)
        else:
            return input.size(0)
