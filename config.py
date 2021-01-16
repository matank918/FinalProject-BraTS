# -*- coding: utf-8 -*-
"""
Created on Wed May 27 17:19:37 2020

@author: matan kachel
"""
import configparser
config = configparser.ConfigParser()

#paths
config['paths'] = {'in_data_path' :'./data/movie17/',
                   'out_data_path_seg' : './data/movie17_seg/',
                   'out_data_path_of' : './data/movie17_of/',
                   'out_data_path_tools' : './data/movie17_tools/'}

#parameters
config['parameters'] = {'in_height' : 1080,
                        'in_width' : 1920,
                        'padded_width' : 'in_width',
                        'padded_height' : 1536,
                        'resize_factor' : 3,
                        'gpu_str' : "cuda",
                        'cpu_str' : "cpu",
                        'semantic_segmentation_factor' : 127,
                        'random_state' : 42,
                        'frame_resize_factor' : 2.828}


#thresholds
config['thresholds'] = {'tool_size_thresh' : 5000,
                        'tool_geo_dist_thresh' : 100,
                        'tool_of_dist_thresh' : 1,
                        'tool_angle_thresh' : 15,
                        'affinity_thresh' : 1.5,
                        'tool_max_val' : 1000,
                        'tool_shape_contour_ratio_thresh' : 1.25,
                        'tool_reliability_thresh' : 0.75,
                        'min_tool_area' : 2500,
                        'curve_sampling_factor' : 5,
                        'min_n_components' : 1,
                        'max_split_tools' : 4,
                        'min_prob' : 0.98,
                        'min_tool_shape' : 50,
                        'max_n_tools' : 5,
                        'min_tool_size' : 1200,
                        'num_of_init_frames' : 3,
                        'num_out_of_frame' : 3,
                        'max_tip_distance' : 150,
                        'end_init_mode' : 3,
                        'remove_from_list' : 3,
                        'split_based_on_tracker' : 0.1,
                        'connect_based_on_tracker' : 0.1,
                        'min_tip_dist' : 50}


#split thresholds
config['split_thresholds'] = {'split_contour_points_thresh': 10,
                              'split_contour_convexity_defect_dist': 10000,
                              'split_contour_leftover_dist': -20,
                              'split_contour_leftover_shape': 10}

#update tools list thresholds
config['update_tools_list_thresholds'] = {'new_tool_reliability_thresh': 0.8,
                                          'update_from_seg_reliability': 0.8,
                                          'update_from_seg_dice': 0.9,

                                          'update_from_tracker_dice': 0.8,
                                          'num_of_frames_delay': 2,
                                          'reduction_level': 0.05,
                                          'tip_reliability_thresh': 0.95,
                                          'segmentation_filter_thresh': 0.9,
                                          'update_tip_loc_old_thresh': 0.9,
                                          'update_tip_loc_new_thresh': 0.1,
                                          'min_tip_reliability':0.2}


# models
config['models'] = {'segmentation_model_path' : 'G:/My Drive/TRX_DL_DataBase/Model/Binary_and_semantic_network/model_0_ep_40.pt'}


#global params
config ['backwarp_tenGrid'] = {}



with open('cfg_file.ini', 'w') as configfile:
    config.write(configfile)