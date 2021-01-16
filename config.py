# -*- coding: utf-8 -*-
"""
Created on Wed May 27 17:19:37 2020

@author: matan kachel
"""
import configparser
config = configparser.ConfigParser()

#model
config['model'] = {}

#loss
config['loss'] = {}


#loader
config['loader'] = {'path':'new data', 'batch size':1, 'val percent':0.1}


#optimizer
config['optimizer'] = {}

#train
config['train'] = {}


with open('cfg_file.ini', 'w') as configfile:
    config.write(configfile)