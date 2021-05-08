# -*- coding: utf-8 -*-
"""
Created on Wed May 27 17:19:37 2020

@author: matan kachel
"""

# general
log_path = r'./runs'
id = 10000
run_name = "Dry runs"
run_purpose = ""

# model
model_name = 'UNet3D'
module_name = 'nnUnet.BuildingBlocks'
basic_block = 'DoubleConv'
in_channels = 4
out_channels = 4
f_maps = [32, 64, 128, 256, 320, 320]
apply_pooling = False

# loss and eval
loss_name = 'DiceLoss'
# loss_name = 'BCEDiceLoss'
# loss_name = 'GeneralizedDiceLoss'
eval_name = 'DiceMetric'

# loader
# loader_path = r'C:\Users\User\Documents\FinalProject\MICCAI_BraTS2020\MICCAI_BraTS2020_TrainingData'
loader_path = '/tcmldrive/shared/BraTS2020 Training/'
val_percent = 0.1
batch_size = 2

# optimizer
deep_supervision = 3
initial_lr = 0.005
accumulation_steps = 1
lr_scheduler_patience = 30
lr_scheduler_eps = 1e-3
weight_decay = 3e-5
# train
max_num_epochs = 15

best_eval_score = 0.65

# architecture
kernel_size = (3, 3, 3)
padding = (1, 1, 1)
stride = (2, 2, 2)
output_padding = (1, 1, 1)
