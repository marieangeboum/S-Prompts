import os
import json
import glob
import copy
import torch
import torch.nn as nn
import time as t
import wandb
import numpy as np
import pandas as pd
import random

import logging
import traceback
import rasterio
import argparse

import seaborn as sns
import matplotlib.pyplot as plt

from PIL import Image
from torch.utils.data import Dataset
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from methods.base import BaseLearner
from torchvision import datasets, transforms
from utils.toolkit import tensor2numpy, accuracy_domain
from models.vitnet import ViTNet

from datetime import datetime, time
from argparse import ArgumentParser
from torchmetrics import ConfusionMatrix, MetricCollection, Accuracy, Precision, Recall

from torch.utils.data import DataLoader, RandomSampler, ConcatDataset
from transformers import ViTFeatureExtractor, ViTModel
from sklearn.metrics import accuracy_score

wandb.login(key="a60322f26edccc6c3f79accc480d56e52e02750a")
# Configure the logging settings
logging.basicConfig(filename='error_log.txt', level=logging.ERROR, format='%(asctime)s - %(levelname)s: %(message)s')
parser = ArgumentParser()
# Préparation du jeu de données
parser.add_argument("--metadata_file", type=str, default = '/scratcht/FLAIR_1/flair-one_classif_metadata.json')
parser.add_argument("--seed", type=int, default= 77)
parser.add_argument("--data_path", type=str, default ='/scratcht/FLAIR_1/train')
parser.add_argument("--mode", type=str, default ='train3')
parser.add_argument('--ceil', type = float, default = 50)
parser.add_argument('--num_gpu', type = str, default = "3")
args = parser.parse_args()

def load_params():
    parser = argparse.ArgumentParser(description='Reproduce of multiple continual learning algorithms.')
    parser.add_argument('--config', type=str, default="configs/eurosat.json",
                        help='Json file of settings.')
    args = parser.parse_args()
    with open(args.config) as data_file:
        param = json.load(data_file)
    _args = vars(args)  # Converting argparse Namespace to a dict.
    _args.update(param)  # Add parameters from json
    return _args

_args = load_params()
EPSILON = _args["EPSILON"]
init_epoch = _args["init_epoch"]
init_lr = _args["init_lr"]
init_lr_decay = _args["init_lr_decay"]
init_weight_decay = _args["init_weight_decay"]
epochs = _args["epochs"]
lrate = _args["lrate"]
lrate_decay = _args["lrate_decay"]
batch_size = _args["batch_size"]
weight_decay = _args["weight_decay"]
num_workers = _args["num_workers"]
tasks_names = _args["task_name"]
class_num = _args["init_cls"]

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = _args["seed"]
topk = 1 # origin is 5

eurosat_classnames = {
    0 : "AnnualCrop",
    1 : "Forest",
    2 : "HerbaceousVegetation", 
    3 : "Highway",
    4 : "Industrial",
    5:  "Pasture", 
    6 : "PermanentCrop",
    7 : "Residential",
    8 : "River", 
    9 : "SeaLake"
    }

# Transformations
test_trsf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(256, antialias=True),
    transforms.CenterCrop(224),
])

# Data Processing
def rasterio_loader(path):
    with rasterio.open(path, 'r') as f:
        image_rasterio = f.read((1,2,3), out_dtype=np.uint8)
    # Convert the NumPy array to a PIL image
    image_rasterio_transposed = np.transpose(image_rasterio, (1, 2, 0))
    image = Image.fromarray(image_rasterio_transposed) 
    return image

# Load Dataset
class eurosatDataset(Dataset):
    # Custom dataset for the data loader
    def __init__(self, txt_file, trsf):
        with open(txt_file, 'r') as f:
            lines = f.read().splitlines()
        self.data = []
        for line in lines:
            path, label = line.split("\t")
            self.data.append((path, label))
        self.trsf = trsf
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path, label = self.data[idx]
        image = self.trsf(rasterio_loader(path))
        
        return idx, image, label, path
