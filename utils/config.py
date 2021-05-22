# -*- coding: utf-8 -*-
"""
Created on Wed May 27 17:19:37 2020

@author: matan kachel
"""

# general
log_path = r'./runs'
id = 2
run_name = "new baseline ???"
run_purpose = "trying to compare between accumulation and batch size"

# model
in_channels = 4
out_channels = 4
f_maps = (32, 64, 128, 256, 320)

# loss
loss_name = 'DiceCE'
# loss_name = 'Dice'
# loss_name = 'GeneralizedDiceLoss'

# loader
loader_path = r'C:\Users\User\Documents\FinalProject\MICCAI_BraTS2020\MICCAI_BraTS2020_TrainingData'
# loader_path = '/tcmldrive/shared/BraTS2020 Training/'
val_percent = 0.2
batch_size = 2

# optimizer
optimizer_name = "Adam"
initial_lr = 3e-4
momentum = 0.99
weight_decay = 3e-5
nesterov = True
amsgrad = True

# train
max_num_epochs = 10
best_eval_score = 0.7
validate_after_iter = 75
log_after_iter = 75

# architecture
kernel_size = (3, 3, 3)
padding = (1, 1, 1)
stride = (2, 2, 2)
output_padding = (1, 1, 1)
