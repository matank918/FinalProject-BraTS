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

#loss and eveal
loss_name = 'DiceLoss'
# loss_name = 'BCEDiceLoss'
eval_name = 'DiceMetric'

#loader
loader_path =r'/tcmldrive/databases/Public/MICA BRaTS2018'
val_percent = 0.1
batch_size = 2

#optimizer
learning_rate = 0.01
momentum = 0.99
nesterov = True

# scheduler
end_learning_rate = 1e-6
max_decay_steps = 1000
power = 4

#train
num_epoch = 1
validate_after_iters = 100
validate_iters = 35
log_after_iters = 50
max_num_epochs = 100
eval_score_higher_is_better = True
num_iterations = 1
checkpoint_dir = r'/home/kachel/project/checkpoint/'
max_num_iterations = 5000
best_eval_score = 0