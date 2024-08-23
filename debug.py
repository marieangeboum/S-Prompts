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
parser.add_argument("--metadata_file", type=str, default = '/run/user/108646/gvfs/sftp:host=baymax/scratcht/FLAIR_1/flair-one_classif_metadata.json')
parser.add_argument("--data_path", type=str, default ='/run/user/108646/gvfs/sftp:host=baymax/scratcht/FLAIR_1/train')
parser.add_argument("--seed", type=int, default= 77)
parser.add_argument("-txt_files", type=str)

parser.add_argument("--mode", type=str, default ='train3')
parser.add_argument('--ceil', type = float, default = 50)
parser.add_argument('--num_gpu', type = str, default = "3")
args = parser.parse_args()

def load_params():
    parser = argparse.ArgumentParser(description='Reproduce of multiple continual learning algorithms.')
    parser.add_argument('--config', type=str, default="configs/flair2_test_sip.json",
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

dir_ = _args["logs_directory"]

train_trsf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(224, antialias=True),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=63/255)
])

test_trsf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(256, antialias=True),
    transforms.CenterCrop(224),
])
def rasterio_loader(path):
    with rasterio.open(path, 'r') as f:
        image_rasterio = f.read((1,2,3), out_dtype=np.uint8)
    # Convert the NumPy array to a PIL image
    image_rasterio_transposed = np.transpose(image_rasterio, (1, 2, 0))
    image = Image.fromarray(image_rasterio_transposed) 
    return image

class DummyDataset(Dataset):
    # Custom dataset for the data loader
    def __init__(self, txt_file, trsf):
        with open(txt_file, 'r') as f:
            lines = f.read().splitlines()
        self.data = []
        for line in lines:
            path, label, bool_ = line.split("\t")
            self.data.append((path, label, bool_))
        self.trsf = trsf
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path, label, bool_ = self.data[idx]
        # image = self.trsf(rasterio_loader(path))
        image = self.trsf(rasterio_loader(path.replace('/scratcht/', '/run/user/108646/gvfs/sftp:host=baymax/scratcht/')))
        return idx, image, label, bool_, path.split("/")[-1].strip(".tif")
    
    
    
    
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
str_seed = str(seed)
print(f"Seed: {str_seed}")
torch.autograd.set_detect_anomaly(True) 
ceil = 50
    
# Définition de la séquence de tâche à traiter
# _cur_task = 0 
_network = ViTNet(_args).to(_device)
all_train_paths = glob.glob(f'/run/user/108646/gvfs/sftp:host=baymax/scratcht/FLAIR_1/flair_txt_files/*_train3_{seed}_{ceil}_.txt')
all_test_paths = glob.glob(f'/run/user/108646/gvfs/sftp:host=baymax/scratcht/FLAIR_1/flair_txt_files/*_test3_{seed}_{ceil}_.txt')

train_dataset = []
test_dataset = []

for train in all_train_paths[:2] : 
    train_dataset.append(DummyDataset(txt_file=train, trsf=train_trsf))
train_dataset = ConcatDataset(train_dataset)
for test in all_test_paths[:2] :
    test_dataset.append(DummyDataset(txt_file=test, trsf=test_trsf))
test_dataset = ConcatDataset(test_dataset)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                               num_workers=num_workers)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                               num_workers=num_workers)
# Définition du _network
class_num = _network.class_num
for name, param in _network.named_parameters():
    param.requires_grad_(False)
    if "classifier" in name:
        param.requires_grad_(True)
# if _cur_task==0:
optimizer = optim.SGD(_network.parameters(), momentum=0.9,lr=init_lr,weight_decay=init_weight_decay)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,T_max=init_epoch)
run_epoch = init_epoch

best_accuracy = 0.0  # Variable to keep track of the best accuracy
prog_bar = tqdm(range(run_epoch))
#examples = []
# metric = ConfusionMatrix(task = 'multiclass',num_classes=class_num,normalize = 'all').to(_device)
# metrics = MetricCollection({'micro_recall': Recall(task = 'multiclass',num_classes=class_num, average='micro'), 'macro_recall': Recall(task = 'multiclass',num_classes=class_num,average='macro'), 'micro_precision': Precision(task = 'multiclass',num_classes=class_num, average='micro'),'macro_precision': Precision(task = 'multiclass',num_classes=class_num, average='macro')}).to(_device)

labels = [
    "building", "pervious surface","impervious surface",
    "bare soil","water","coniferous",  "deciduous",
    "brushwood", "vineyard", "herbaceous vegetation",
    "agricultural land", "plowed land", "other" ]
for _, epoch in enumerate(prog_bar):
    _network.train()  # Set the network to training mode
    losses = 0.
    correct, total = 0, 0
    for i, ( _,inputs, targets, bool_, img_name) in enumerate(train_loader):
        inputs = inputs.requires_grad_().to(_device)
        targets = torch.tensor([int(t) for t in targets]).to(_device)
        logits = _network(inputs).to(_device)
        loss = F.cross_entropy(logits, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses += loss.item()
        probabilities = F.softmax(logits, dim=1)
        _, preds = torch.max(probabilities, dim=1)
        correct += preds.eq(targets).sum().item()
        total += targets.size(0)
