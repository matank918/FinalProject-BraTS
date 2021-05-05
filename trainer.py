import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from utils.utils import RunningAverage, split_image
import os
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
        checkpoint_dir (string): dir for saving checkpoints and tensorboard logs
        max_num_epochs (int): maximum number of epochs
        max_num_iterations (int): maximum number of iterations
        validate_after_iters (int): validate after that many iterations
        log_after_iters (int): number of iterations before logging to tensorboard
    """

    def __init__(self, model, logger, optimizer, loss_criterion, lr_scheduler,
                 eval_criterion, device, checkpoint_dir, best_eval_score,
                 validate_after_iters, log_after_iters, max_num_iterations,
                 max_num_epochs, accumulation_steps):

        self.logger = logger
        self.writer = SummaryWriter()
        self.model = model
        self.optimizer = optimizer
        self.scheduler = lr_scheduler
        self.accumulation_steps = accumulation_steps
        self.loss_criterion = loss_criterion
        self.eval_criterion = eval_criterion
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.max_num_epochs = max_num_epochs
        self.max_num_iterations = max_num_iterations
        self.validate_after_iters = validate_after_iters
        self.log_after_iters = log_after_iters

        self.logger.info(model)

        if best_eval_score is not None:
            self.best_eval_score = best_eval_score
        else:
            # initialize the best_eval_score
            self.best_eval_score = float('-inf')

    def fit(self, trainloaders, evalloader):
        for num_epoch in range(1, self.max_num_epochs + 1):
            train_losses = RunningAverage()
            train_eval_scores = RunningAverage()

            # sets the model in training mode
            self.model.train()
            for i, batch in enumerate(trainloaders, start=1):

                iter_time = time.clock()

                input, target = self._split_batch(batch)

                print(f'Training iteration [{i}/{self.max_num_iterations}]. '
                      f'Epoch [{num_epoch}/{self.max_num_epochs}]')

                # get output from the net, calculate the loss
                output, loss = self._forward_pass(input, target)
                train_losses.update(loss.item(), self._batch_size(input))

                # compute eval criterion for training set
                eval_score = self.eval_criterion(output, target)
                train_eval_scores.update(eval_score.item(), self._batch_size(input))

                loss = loss / self.accumulation_steps
                loss.backward()

                # compute gradients and update parameters
                if i % self.accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                if i % self.validate_after_iters == 0:
                    global_step = num_epoch * len(trainloaders) + i

                    # evaluate during training
                    if evalloader is not None:
                        self.model.eval()
                        # evaluate on validation set
                        val_losses, eval_score = self.validate(evalloader)
                        self._log_stats(global_step, 'val', val_losses, eval_score)

                        # set the model back to training mode
                        self.model.train()

                        # remember best validation metric
                        is_best = self._is_best_eval_score(eval_score)
                        if is_best:
                            self._save_checkpoint(i, num_epoch)

                if i % self.log_after_iters == 0:
                    global_step = num_epoch * len(trainloaders) + i

                    # log stats, params and images
                    self.logger.info(
                        f'Training stats. Loss: {train_losses.avg}. Evaluation score: {train_eval_scores.avg}')

                    self._log_stats(global_step, 'train', train_losses.avg, train_eval_scores.avg)
                    self._log_params(global_step)
                    # self._log_images(input, global_step, target, output, 'train_')

                iter_time = time.clock() - iter_time
                self.logger.info(f"duration for iter {i} is {iter_time} seconds")

            # adjust learning rate if necessary
            if self.scheduler is not None:
                global_step = num_epoch * len(trainloaders)
                self.scheduler.step()
                # log current learning rate in tensorboard
                self._log_lr(global_step)

            self.logger.info(f"Reached maximum number of epochs: {self.max_num_epochs}. Finishing training...")

    def validate(self, evalloader):

        print('Validating...')

        val_losses = RunningAverage()
        val_scores = RunningAverage()

        with torch.no_grad():
            for i, batch in enumerate(evalloader):
                input, target = self._split_batch(batch)

                output, loss = self._forward_pass(input, target)
                val_losses.update(loss.item(), self._batch_size(input))

                eval_score = self.eval_criterion(output, target)
                val_scores.update(eval_score.item(), self._batch_size(input))

            self.logger.info(f'Validation finished. Loss: {val_losses.avg}. Evaluation score: {val_scores.avg}')
            return val_losses.avg, val_scores.avg

    def _split_batch(self, batch):
        # splits between input and target
        input, target = batch
        input = input.to(self.device)
        target = target.to(self.device)
        return input, target

    def _forward_pass(self, input, target):
        output = self.model(input)
        fin_output = output[-1]
        # compute the loss
        loss = self.loss_criterion(fin_output, target)
        if self.model.training:
            for layer_output in output[:-1]:
                loss += self.loss_criterion(layer_output, target)

        output = output[-1]

        return output, loss

    def _is_best_eval_score(self, eval_score):
        is_best = eval_score > self.best_eval_score

        if is_best:
            self.logger.info(f'Saving new best evaluation metric: {eval_score}')
            self.best_eval_score = eval_score

        return is_best

    def _save_checkpoint(self, num_iterations, num_epoch):
        if isinstance(self.model, nn.DataParallel):
            state_dict = self.model.module.state_dict()
        else:
            state_dict = self.model.state_dict()

        state = {'epoch': num_epoch + 1,
                 'num_iterations': num_iterations,
                 'model_state_dict': state_dict,
                 'best_eval_score': self.best_eval_score,
                 'optimizer_state_dict': self.optimizer.state_dict(),
                 'device': str(self.device),
                 'max_num_epochs': self.max_num_epochs,
                 'max_num_iterations': self.max_num_iterations,
                 'validate_after_iters': self.validate_after_iters,
                 'log_after_iters': self.log_after_iters,
                 }

        if not os.path.exists(self.checkpoint_dir):
            self.logger.info(
                f"Checkpoint directory does not exists. Creating {self.checkpoint_dir}")
            os.mkdir(self.checkpoint_dir)

        last_file_path = os.path.join(self.checkpoint_dir, 'last_checkpoint.pytorch')
        self.logger(f"Saving last checkpoint to '{last_file_path}'")
        torch.save(state, last_file_path)

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
