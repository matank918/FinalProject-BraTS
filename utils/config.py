# -*- coding: utf-8 -*-
"""
Created on Wed May 27 17:19:37 2020

@author: matan kachel
"""

# general
log_path = r'./runs'
id = 3
run_name = "Dry run3 -gradient accumulation verification"
run_purpose = "compare gradient accumulation to batch size"

# model
model_name = 'UNet3D'
module_name = 'nnUnet.BuildingBlocks'
basic_block = 'DoubleConv'
in_channels = 4
out_channels = 4
f_maps = [32, 64, 128, 256, 320, 320]
apply_pooling = False

# loss and eveal
loss_name = 'DiceLoss'
# loss_name = 'BCEDiceLoss'
# loss_name = 'GeneralizedDiceLoss'
eval_name = 'DiceMetric'

# loader
loader_path = r'C:\Users\User\Documents\FinalProject\MICCAI_BraTS2020\MICCAI_BraTS2020_TrainingData'
val_percent = 0.1
batch_size = 1

# optimizer
deep_supervision = 3
optimizer_name = 'SGD'
# optimizer_name = 'Adam'
learning_rate = 0.01
momentum = 0.99
nesterov = True
accumulation_steps = 2

# train
log_after_iters = 50
validate_after_iters = 50 # < len(traindata)
max_num_epochs = 10

best_eval_score = 0.65
checkpoint_dir = r'./runs'

# architecture
kernel_size = (3, 3, 3)
padding = (1, 1, 1)
stride = (2, 2, 2)
output_padding = (1, 1, 1)
