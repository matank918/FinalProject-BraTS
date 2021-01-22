# -*- coding: utf-8 -*-
"""
Created on Wed May 27 17:19:37 2020

@author: matan kachel
"""
#model
model_name = 'UNet3D'
in_channels = 4
out_channels = 1
f_maps = [32, 64, 128, 256, 320, 320]
apply_pooling = False
interpolate = False
testing = False
#loss
loss_name = 'DiceLoss'

#loader
loader_path =r'C:\Users\User\Documents\FinalProject\MICA BRaTS2018\Training'
val_percent = 0.1
bath_size = 1

#optimizer
learning_rate = 0.01
weight_decay = 0.1

# scheduler
step_size = 30
gamma = 0.1
#train
num_epoch = 0
validate_after_iters = 10
validate_iters = 1
log_after_iters = 5
max_num_epochs = 1
eval_score_higher_is_better = True
skip_train_validation = False
num_iterations = 1
checkpoint_dir = r'C:\Users\User\Documents\FinalProject\FinalProject-BraTS'
max_num_iterations = 1000
best_eval_score = 0