import argparse

import torch
import yaml
import sys


def load_config():

    parser = argparse.ArgumentParser(description='UNet3D')
    parser.add_argument('--config', type=str, help='Path to the YAML config file', required=True)
    args = parser.parse_args()
    config = _load_config_yaml(args.config)
    # Get a device to train on
    return config


def _load_config_yaml(config_file):
    return yaml.safe_load(open(config_file, 'r'))




if __name__ == '__main__':
    load_config()
