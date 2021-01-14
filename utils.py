import logging
import sys
import numpy as np


class RunningAverage:
    """Computes and stores the average
    """

    def __init__(self):
        self.count = 0
        self.sum = 0
        self.avg = 0

    def update(self, value, n=1):
        self.count += n
        self.sum += value * n
        self.avg = self.sum / self.count



def get_logger(name, level=logging.DEBUG):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    # Logging to console
    c_handle = logging.StreamHandler(sys.stdout)
    c_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    c_handle.setFormatter(c_format)
    logger.addHandler(c_handle)
    # logging to file
    f_handler = logging.FileHandler('file.log')
    f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    f_handler.setFormatter(f_format)
    logger.addHandler(f_handler)

    return logger


def get_number_of_learnable_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    return sum([np.prod(p.size()) for p in model_parameters])


