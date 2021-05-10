# -*- coding: utf-8 -*-
"""
Created on Wed May 27 17:19:37 2020

@author: matan kachel
"""

# general
log_path = r'./runs'
id = 1
run_name = "new baseline ??? - monai"
run_purpose = "trying to establish new baseline with monai"

# model
# model_name = 'UNet3D'
# module_name = 'nnUnet.BuildingBlocks'
# # basic_block = 'BatchDoubleConv'
# basic_block = 'DoubleConv'
# in_channels = 4
# out_channels = 4
# f_maps = [32, 64, 128, 256, 320, 320]
# apply_pooling = False

# loss and eval
loss_name = 'DiceCE'
# loss_name = 'BCEDiceLoss'
# loss_name = 'GeneralizedDiceLoss'
eval_name = 'DiceMetric'

# loader
loader_path = r'C:\Users\User\Documents\FinalProject\MICCAI_BraTS2020\MICCAI_BraTS2020_TrainingData'
# loader_path = '/tcmldrive/shared/BraTS2020 Training/'
val_percent = 0.1
batch_size = 1
accumulation_steps = 2

# optimizer
optimizer_name = "Adam"
deep_supervision = 0
initial_lr = 3e-4
momentum = 0.99
weight_decay = 3e-5
nesterov = True
amsgrad = True

include_background = True
# train
max_num_epochs = 10
best_eval_score = 0.7
validate_after_iter = 50
log_after_iter = 50
# architecture
# kernel_size = (3, 3, 3)
# padding = (1, 1, 1)
# stride = (2, 2, 2)
# output_padding = (1, 1, 1)
