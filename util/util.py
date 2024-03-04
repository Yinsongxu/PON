import numpy as np
import logging
import os
import time
import torch
import shutil
import os


def setup_logger(logger_name, root, level=logging.INFO, screen=False, tofile=False):
    """set up logger"""
    lg = logging.getLogger(logger_name)
    formatter = logging.Formatter("[%(asctime)s.%(msecs)03d] %(message)s", datefmt="%H:%M:%S")
    lg.setLevel(level)
    if tofile:
        log_file = os.path.join(root, "{}_{}.log".format(logger_name, get_timestamp()))
        fh = logging.FileHandler(log_file, mode="w")
        fh.setFormatter(formatter)
        lg.addHandler(fh)
    if screen:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        lg.addHandler(sh)
    return lg


def get_timestamp():
    timestampTime = time.strftime("%H%M%S")
    timestampDate = time.strftime("%Y%m%d")
    return timestampDate + "-" + timestampTime


def save_checkpoint(state, is_best, checkpoint):
    filepath_last = os.path.join(checkpoint, "last.pth.tar")
    filepath_best = os.path.join(checkpoint, "best.pth.tar")
    if not os.path.exists(checkpoint):
        print("Checkpoint Directory does not exist! Masking directory {}".format(checkpoint))
        os.mkdir(checkpoint)
    else:
        print("Checkpoint DIrectory exists!")
    torch.save(state, filepath_last)
    if is_best:
        if os.path.isfile(filepath_best):
            os.remove(filepath_best)
        shutil.copyfile(filepath_last, filepath_best)
