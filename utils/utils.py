import os
import glob
import torch
import sys
import shutil
import random
import logging
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from gymnasium.spaces import MultiDiscrete, Discrete, Box
from moviepy.editor import VideoFileClip, concatenate_videoclips


def config_logging(log_file="main.log", log_level=logging.INFO):
    date_format = '%Y-%m-%d %H:%M:%S'
    log_format = '%(asctime)s: [%(levelname)s]: %(message)s'
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(logging.Formatter(log_format, datefmt=date_format))

    # Set up the FileHandler for logging to a file
    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setFormatter(logging.Formatter(log_format, datefmt=date_format))
    logging.basicConfig(level=log_level, handlers=[stdout_handler, file_handler])
    
def set_seed_everywhere(env: gym.Env, seed=0):
    env.action_space.seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # pl.seed_everything(cfg.seed) #equal to fist 3 lines and it can pass seeds
    
def get_confusionMat(pred, target, n_class):
    #y row: pred, x column: target
    if isinstance(pred, torch.Tensor) and isinstance(target, torch.Tensor):
        with torch.no_grad():
            # _, pred = output.topk(1, 1, True, True)
            # pred = pred.t()
            confusionMat = torch.zeros((n_class, n_class))
            for i in range(n_class):
                for j in range(n_class):
                    confusionMat[i, j] = torch.sum((pred==i) & (target==j))
            return confusionMat
    else:
        confusionMat = np.zeros((n_class, n_class), dtype=int)
        for i in range(n_class):
            for j in range(n_class):
                confusionMat[i, j] = np.sum((pred == i) & (target == j))
        return confusionMat