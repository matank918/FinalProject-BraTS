# -*- coding: utf-8 -*-
"""
Created on Wed May 27 17:19:37 2020

@author: matan kachel
"""
#model
model_name = 'UNet3D'
module_name = 'BuildingBlocks'
basic_block = 'DoubleConv'
in_channels = 4
out_channels = 4
f_maps = [32, 64, 128, 256, 320, 320]
apply_pooling = False
interpolate = True
testing = True

#loss and eveal
loss_name = 'DiceLoss'
# loss_name = 'BCELoss'
eval_name = 'DiceMetric'

#loader
loader_path =r'C:\Users\User\Documents\FinalProject\MICA BRaTS2018\Training'
val_percent = 0.1
batch_size = 1

#optimizer
learning_rate = 0.1
momentum = 0.99
nesterov = True
# scheduler
step_size = 30
gamma = 0.1


#train
num_epoch = 1
validate_after_iters = 10
validate_iters = 20
log_after_iters = 30
max_num_epochs = 100
eval_score_higher_is_better = True
skip_train_validation = True
num_iterations = 1
checkpoint_dir = r'C:\Users\User\Documents\FinalProject\FinalProject-BraTS'
max_num_iterations = 400
best_eval_score = 0