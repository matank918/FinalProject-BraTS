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
interpolate = True
testing = False

#loss and eveal
loss_name = 'DiceLoss'
eval_name = 'DiceLoss'

#loader
loader_path =r'C:\Users\User\Documents\FinalProject\MICA BRaTS2018\Training'
val_percent = 0.1
batch_size = 1

#optimizer
learning_rate = 0.01
weight_decay = 0.1

# scheduler
step_size = 30
gamma = 0.1


#train
num_epoch = 0
validate_after_iters = 10
validate_iters = 3
log_after_iters = 10
max_num_epochs = 10
eval_score_higher_is_better = False
skip_train_validation = False
num_iterations = 1
checkpoint_dir = r'C:\Users\User\Documents\FinalProject\FinalProject-BraTS'
max_num_iterations = 3000
best_eval_score = 0