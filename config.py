# -*- coding: utf-8 -*-
"""
Created on Wed May 27 17:19:37 2020

@author: matan kachel
"""
import configparser
config = configparser.ConfigParser()

# general
config['general'] = {'name':'UNet3D'}

#model
config['model'] = {'in_channels': 4, 'out_channels':4, 'f_maps':[32, 64, 128, 256, 320, 320],
                   'apply_pooling': False, 'interpolate':False,'testing':False}

#loss
config['loss'] = {}


#loader
config['loader'] = {'path':'Data', 'batch size':1}


#optimizer
config['optimizer'] = {}

#train
config['train'] = {}


with open('cfg_file.ini', 'w') as configfile:
    config.write(configfile)