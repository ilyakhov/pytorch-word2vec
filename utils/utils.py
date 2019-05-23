import logging
import json
import os
import argparse


def set_logger(log_path):
    """Sets the logger to log info in terminal and file `log_path`.
    In general, it is useful to have a logger so that every output
    to the terminal is saved in a permanent file.
    Here we save it to `model_dir/train.log`.
    Example:
    ```
    logging.info("Starting training...")
    ```
    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)


class Params:
    def __init__(self, jsonpath=None):
        self.load(jsonpath)

    def load(self, path):
        params = json.load(open(path))
        self.__dict__.update(params)
        if getattr(self, 'downsampling_params', None) is not None:
            setattr(self, 'downsampling_params',
                    tuple(self.downsampling_params))


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def make_directory(path):
    path = os.path.abspath(path)
    if os.path.exists(path):
        return 'Path {} exists!'.format(path)
    else:
        splits = path.split('/')
        newpath = '/'
        for i, split in enumerate(splits):
            if split == '': continue
            newpath = os.path.join(newpath, split)
            if os.path.exists(newpath): continue
            else:
                os.mkdir(newpath)
        if os.path.exists(path):
            return 'Path {} created!'.format(path)
        else:
            raise FileExistsError('Path {} creation is failed!'.format(path))
