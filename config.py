# -*- coding: utf-8 -*-
"""
Created on Wed May 27 17:19:37 2020

@author: matan kachel
"""
import configparser
config = configparser.ConfigParser()

#model
config['model'] = {'name':'UNet3D','in_channels': 4, 'out_channels':1, 'f_maps':[32, 64, 128, 256, 320, 320],
                   'apply_pooling': False, 'interpolate':False,'testing':False}

#loss
config['loss'] = {'name':'DiceLoss'}


#loader
config['loader'] = {'path':r'C:\Users\User\Documents\FinalProject\MICA BRaTS2018\Training', 'val_percent':0.1, 'batch size':1}


#optimizer
config['optimizer'] = {'learning_rate':0.01, 'weight_decay':0.1}

#train
config['train'] = {'validate_iters':20,'skip_train_validation':False,'iters':1, 'validate_after_iters':1,'log_after_iters':False}


with open('cfg_file.ini', 'w') as configfile:
    config.write(configfile)