# -*- coding: utf-8 -*-
"""
Created on Wed May 27 17:19:37 2020

@author: matan kachel
"""

# general
log_path = r'/home/kachel/Project/runs/'

#model
model_name = 'UNet3D'
module_name = 'BuildingBlocks'
basic_block = 'DoubleConv'
in_channels = 4
out_channels = 4
f_maps = [32, 64, 128, 256, 320, 320]
apply_pooling = False

#loss and eveal
# loss_name = 'DiceLoss'
loss_name = 'BCEDiceLoss'
# loss_name = 'GeneralizedDiceLoss'
eval_name = 'DiceMetric'

#loader
loader_path =r'/home/kachel/MICA BraTS2020/'
val_percent = 0.1
batch_size = 2

#optimizer
optimizer_name = 'SGD'
# optimizer_name = 'Adam'
learning_rate = 0.01
momentum = 0.99
nesterov = True

#train
log_after_iters = 90
validate_after_iters = 90
validate_iters = 40

max_num_epochs = 50
max_num_iterations = 5000

best_eval_score = 0
eval_score_higher_is_better = True
checkpoint_dir = r'/home/kachel/Project/checkpoint/'
num_epoch = 1
num_iterations = 1
