import logging
import sys
from datetime import date, datetime
import os
import socket


def get_logger(path, level=logging.INFO):
    log_name = get_log_name()
    cwd = os.getcwd()
    os.chdir(path)
    logging.basicConfig(filename=log_name, level=level)
    logger = logging.getLogger()

    # Logging to console
    c_handle = logging.StreamHandler(sys.stdout)
    c_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    c_handle.setFormatter(c_format)
    logger.addHandler(c_handle)

    # logging to file
    f_handler = logging.FileHandler(log_name)
    f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    f_handler.setFormatter(f_format)
    logger.addHandler(f_handler)
    os.chdir(cwd)
    return logger


def get_date_and_time():
    today = date.today()

    now = datetime.now()

    current_date = today.strftime("%b%d")
    time = now.strftime("%H-%M-%S")

    data_and_time = current_date + '_' + time

    return data_and_time


def get_log_name():
    date_and_time = get_date_and_time()
    cp_name = socket.gethostname()
    log_name = date_and_time + '_' + cp_name + '.log'
    return log_name


def get_module_variable(module):
    module = module
    book = {}
    if module:
        book = {key: value for key, value in module.__dict__.items() if not (key.startswith('__') or key.startswith('_'))}
    return book
