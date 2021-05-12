import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from utils.utils import RunningAverage, split_image
import os
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau
from monai.transforms import Activations, AsDiscrete, Compose
from monai.networks.utils import one_hot


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
        checkpoint_dir (string): dir for saving checkpoints and tensorboard logs
        max_num_epochs (int): maximum number of epochs
    """

    def __init__(self, model, logger, optimizer, loss_criterion, lr_scheduler,
                 eval_criterion, device, best_eval_score, log_after_iter, validate_after_iter,
                 max_num_epochs, accumulation_steps, checkpoint_dir=None):

        self.logger = logger
        self.writer = SummaryWriter()
        self.model = model
        self.log_after_iter = log_after_iter
        self.validate_after_iter = validate_after_iter
        self.optimizer = optimizer
        self.scheduler = lr_scheduler
        self.accumulation_steps = accumulation_steps
        self.loss_criterion = loss_criterion
        self.eval_criterion = eval_criterion
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.max_num_epochs = max_num_epochs
        self.logger.info(model)

        if best_eval_score is not None:
            self.best_eval_score = best_eval_score
        else:
            # initialize the best_eval_score
            self.best_eval_score = float('-inf')

    def train(self, trainloaders, evalloader):
        for num_epoch in range(1, self.max_num_epochs + 1):
            train_losses = RunningAverage()
            train_eval_scores = RunningAverage()
            epoch_time = time.time()

            # sets the model in training mode
            self.model.train()
            for i, batch in enumerate(trainloaders, start=1):

                iter_time = time.time()
                input, target = self._split_batch(batch)

                # get output from the net, calculate the loss
                output, loss = self._forward_pass(input, target)
                train_losses.update(loss.item())

                # compute eval criterion for training set
                value, not_nans = self.eval_criterion(output, target)
                train_eval_scores.update(torch.mean(value).item(), not_nans)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # compute gradients and update parameters
                global_step = num_epoch * len(trainloaders) + i
                # if i % self.accumulation_steps == 0:
                #     self.optimizer.step()
                #     self.optimizer.zero_grad()
                if self.scheduler is not None:
                    # adjust learning rate if necessary
                    self.scheduler.step()
                    # log current learning rate in tensorboard

                self.logger.info(f'Training iteration [{i}/{len(trainloaders)}]. '
                                 f'Epoch [{num_epoch}/{self.max_num_epochs}]. '
                                 f"train_loss: {loss.item():.4f}")

                if i % self.log_after_iter == 0:
                    # log stats, params and images
                    self.logger.info(
                        f'Average Training stats:'
                        f' Loss: {train_losses.avg:.4f}. '
                        f' Evaluation score: {train_eval_scores.avg:.4f}. '
                    )

                    self._log_stats(global_step, 'train', train_losses.avg, train_eval_scores.avg)
                    self._log_params(global_step)
                    self._log_lr(global_step)
                    # self._log_images(input, global_step, target, output, 'train_')

                if i % self.validate_after_iter == 0 and evalloader is not None:
                    self.model.eval()

                    # evaluate on validation set
                    val_losses, val_score = self.validate(evalloader)
                    self._log_stats(global_step, 'val', val_losses, val_score)

                    # set the model back to training mode
                    self.model.train()

                    # remember best validation metric
                    is_best = self._is_best_eval_score(val_score)
                    if is_best:
                        self._save_checkpoint(num_epoch)

                iter_time = time.time() - iter_time
                self.logger.info(f"Iter {i} duration is {iter_time:.2f} seconds")

            epoch_time = time.time() - epoch_time
            self.logger.info(f"epoch {num_epoch} duration is {(epoch_time / 60):.2f} minutes")

        self.logger.info(f"Reached maximum number of epochs: {self.max_num_epochs}. Finishing training...")

    def validate(self, evalloader):

        print('Validating...')

        val_losses = RunningAverage()
        val_scores = RunningAverage()
        metric_values_tc = RunningAverage()
        metric_values_wt = RunningAverage()
        metric_values_et = RunningAverage()

        with torch.no_grad():
            for i, batch in enumerate(evalloader):
                input, target = self._split_batch(batch)

                output, loss = self._forward_pass(input, target)
                val_losses.update(loss.item())

                # compute overall mean dice
                value, not_nans = self.eval_criterion(y_pred=output, y=target)
                not_nans = not_nans.item()
                val_scores.update(value.item(), not_nans)
                # compute mean dice for TC
                value_tc, not_nans = self.eval_criterion(y_pred=output[:, 0:1], y=target[:, 0:1])
                not_nans = not_nans.item()
                metric_values_tc.update(value_tc.item(), not_nans)
                # compute mean dice for WT
                value_wt, not_nans = self.eval_criterion(y_pred=output[:, 1:2], y=target[:, 1:2])
                not_nans = not_nans.item()
                metric_values_wt.update(value_wt.item(), not_nans)
                # compute mean dice for ET
                value_et, not_nans = self.eval_criterion(y_pred=output[:, 2:3], y=target[:, 2:3])
                not_nans = not_nans.item()
                metric_values_et.update(value_et.item(), not_nans)

            self.logger.info(
                f"current loss: {val_losses.avg:.4f} "f" current mean dice: {val_scores.avg:.4f}"
                f" tc: {metric_values_tc.avg:.4f} wt: {metric_values_wt.avg:.4f} et: {metric_values_et.avg:.4f}"
            )

            return val_losses.avg, val_scores.avg

    def _split_batch(self, batch):
        # splits between input and target
        input, target = batch['image'], batch['seg']
        input = input.to(self.device)
        target = target.to(self.device)

        return input, target

    def _forward_pass(self, input, target):
        post_trans = Compose([
                Activations(sigmoid=False, softmax=True), AsDiscrete(threshold_values=True)
            ])

        output = self.model(input)
        if isinstance(output, list):
            fin_output = output[-1]
            loss = self.loss_criterion(fin_output, target)
            if self.model.training:
                for layer_output in output[:-1]:
                    loss += self.loss_criterion(layer_output, target)

            output = output[-1]
        else:
            loss = self.loss_criterion(output, target)
        output = post_trans(output)
        return output, loss

    def _is_best_eval_score(self, eval_score):
        is_best = eval_score > self.best_eval_score

        if is_best:
            self.logger.info(f'Saving new best evaluation metric: {eval_score:.4f}')
            self.best_eval_score = eval_score

        return is_best

    def _save_checkpoint(self, num_epoch):
        if self.checkpoint_dir is not None:
            if isinstance(self.model, nn.DataParallel):
                state_dict = self.model.module.state_dict()
            else:
                state_dict = self.model.state_dict()

            state = {'epoch': num_epoch + 1,
                     'model_state_dict': state_dict,
                     'best_eval_score': self.best_eval_score,
                     'optimizer_state_dict': self.optimizer.state_dict(),
                     'device': str(self.device),
                     'max_num_epochs': self.max_num_epochs,
                     }

            if not os.path.exists('./runs'):
                os.mkdir('./runs')

            self.logger(f"Saving last checkpoint to '{self.checkpoint_dir}'")
            torch.save(state, self.checkpoint_dir)

    def _log_lr(self, global_step):
        lr = self.optimizer.param_groups[0]['lr']
        self.writer.add_scalar('learning_rate', lr, global_step)

    def _log_stats(self, global_step, phase, loss_avg, eval_score_avg):
        tag_value = {
            f'{phase}_loss_avg': loss_avg,
            f'{phase}_eval_score_avg': eval_score_avg
        }

        for tag, value in tag_value.items():
            self.writer.add_scalar(tag, value, global_step)

    def _log_params(self, global_step):
        for name, value in self.model.named_parameters():
            try:
                self.writer.add_histogram(name, value.data.cpu().numpy(), global_step)
                self.writer.add_histogram(name + '/grad', value.grad.data.cpu().numpy(), global_step)
            except AttributeError as e:
                pass

    def _log_images(self, global_step, input, target, prediction, prefix):
        if self._batch_size(input) == 1:
            _, ch1, ch2, ch4 = torch.chunk(target, dim=1, chunks=4)
            Sagittal_ch1, Coronal_ch1, Horizontal_ch1 = split_image(ch1)
            Sagittal_ch2, Coronal_ch2, Horizontal_ch2 = split_image(ch2)
            Sagittal_ch4, Coronal_ch4, Horizontal_ch4 = split_image(ch4)

            _, pred_ch1, pred_ch2, pred_ch4 = torch.chunk(prediction, dim=1, chunks=4)
            Sagittal_pred_ch1, Coronal_pred_ch1, Horizontal_pred_ch1 = split_image(pred_ch1)
            Sagittal_pred_ch2, Coronal_pred_ch2, Horizontal_pred_ch2 = split_image(pred_ch2)
            Sagittal_pred_ch4, Coronal_pred_ch4, Horizontal_pred_ch4 = split_image(pred_ch2)

            self.writer.add_image(prefix + "seg_Sagittal_ch1", Sagittal_ch1, global_step, dataformats='CHW')
            self.writer.add_image(prefix + "seg_Coronal_ch1", Coronal_ch1, global_step, dataformats='CHW')
            self.writer.add_image(prefix + "seg_Horizontal_ch1", Horizontal_ch1, global_step, dataformats='CHW')
            self.writer.add_image(prefix + "seg_Sagittal_ch2", Sagittal_ch2, global_step, dataformats='CHW')
            self.writer.add_image(prefix + "seg_Coronal_ch2", Coronal_ch2, global_step, dataformats='CHW')
            self.writer.add_image(prefix + "seg_Horizontal_ch2", Horizontal_ch2, global_step, dataformats='CHW')
            self.writer.add_image(prefix + "seg_Sagittal_ch4", Sagittal_ch4, global_step, dataformats='CHW')
            self.writer.add_image(prefix + "seg_Coronal_ch4", Coronal_ch4, global_step, dataformats='CHW')
            self.writer.add_image(prefix + "seg_Horizontal_ch4", Horizontal_ch4, global_step, dataformats='CHW')
            self.writer.add_image(prefix + "pred_Sagittal_ch1", Sagittal_pred_ch1, global_step,
                                  dataformats='CHW')
            self.writer.add_image(prefix + "pred_Coronal_ch1", Coronal_pred_ch1, global_step, dataformats='CHW')
            self.writer.add_image(prefix + "pred_Horizontal_ch1", Horizontal_pred_ch1, global_step,
                                  dataformats='CHW')
            self.writer.add_image(prefix + "pred_Sagittal_ch2", Sagittal_pred_ch2, global_step,
                                  dataformats='CHW')
            self.writer.add_image(prefix + "pred_Coronal_ch2", Coronal_pred_ch2, global_step, dataformats='CHW')
            self.writer.add_image(prefix + "pred_Horizontal_ch2", Horizontal_pred_ch2, global_step,
                                  dataformats='CHW')
            self.writer.add_image(prefix + "pred_Sagittal_ch4", Sagittal_pred_ch4, global_step,
                                  dataformats='CHW')
            self.writer.add_image(prefix + "pred_Coronal_ch4", Coronal_pred_ch4, global_step, dataformats='CHW')
            self.writer.add_image(prefix + "pred_Horizontal_ch4", Horizontal_pred_ch4, global_step,
                                  dataformats='CHW')

    def _add_noise(self, alfa):
        model_parameters = list(filter(lambda p: p.requires_grad, self.model.parameters()))
        for param in model_parameters:
            noise = torch.randn(param.data.shape).to(self.device)
            param.data += noise * alfa

    @staticmethod
    def _batch_size(input):
        if isinstance(input, list) or isinstance(input, tuple):
            return input[0].size(0)
        else:
            return input.size(0)
