import os
import json
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
from torchmetrics.classification import MulticlassConfusionMatrix
from torchmetrics import ConfusionMatrix, MetricCollection, Accuracy, Precision, Recall
from torch.utils.data import DataLoader, RandomSampler, ConcatDataset
from transformers import ViTFeatureExtractor, ViTModel
from sklearn.metrics import accuracy_score,  confusion_matrix, ConfusionMatrixDisplay


def load_params():
    parser = argparse.ArgumentParser(description='Reproduce of multiple continual learning algorithms.')
    parser.add_argument('--config', type=str, default="configs/flair2_5_sip.json",
                        help='Json file of settings.')
    args = parser.parse_args()
    with open(args.config) as data_file:
        param = json.load(data_file)
    _args = vars(args)  # Converting argparse Namespace to a dict.
    _args.update(param)  # Add parameters from json
    return _args

wandb.login(key="a60322f26edccc6c3f79accc480d56e52e02750a")
logging.basicConfig(filename='error_log.txt', level=logging.ERROR, format='%(asctime)s - %(levelname)s: %(message)s')
parser = ArgumentParser()
parser.add_argument("--metadata_file", type=str, default = '/scratcht/FLAIR_1/flair-one_classif_metadata.json')
parser.add_argument("--seed", type=int, default= 77)
parser.add_argument("--data_path", type=str, default ='/scratcht/FLAIR_1/train')
parser.add_argument("--mode", type=str, default ='train2')
parser.add_argument('--ceil', type = float, default = 50)
parser.add_argument('--num_gpu', type = str, default = "3")
args = parser.parse_args()

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
print(class_num)
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = _args["seed"]
topk = 1 
_seed = 77

dir_ = "/inf_logs/"
#if not os.path.exists(dir_):
#    os.makedirs(dir_)

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
        image = self.trsf(rasterio_loader(path))
        
        return idx, image, label, bool_, path.split("/")[-1].strip(".tif")
    
from collections import deque
        
def train_function(_network, train_loader, test_loader, optimizer, scheduler, run_epoch, _cur_task, class_num,ceil, seed, device = _device):
    best_accuracy = 0.0  # Variable to keep track of the best accuracy
    prog_bar = tqdm(range(run_epoch))
    #examples = []
    metric = ConfusionMatrix(task = 'multiclass',num_classes=class_num,normalize = 'all').to(_device)
    metrics = MetricCollection({'micro_recall': Recall(task = 'multiclass',num_classes=class_num, average='micro'),
                                'macro_recall': Recall(task = 'multiclass',num_classes=class_num,average='macro'),
                                'micro_precision': Precision(task = 'multiclass',num_classes=class_num, average='micro'),
                                'macro_precision': Precision(task = 'multiclass',num_classes=class_num, average='macro')}).to(_device)
    
    labels = ["Anthropized", "Natural","Vegetation", "Agricultural","Herbeceous"]
    # labels = [
    #     "building", "pervious surface","impervious surface",
    #     "bare soil","water","coniferous",  "deciduous",
    #     "brushwood", "vineyard", "herbaceous vegetation",
    #     "agricultural land", "plowed land", "other" ]
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
                 
        scheduler.step()
        train_acc = np.around(correct * 100 / total, decimals=2)
        test_acc = _compute_accuracy_domain(model=_network, loader=test_loader)
        test_class_accuracy, test_class_precision, test_class_recall = compute_accuracy_per_class(model=_network, loader=test_loader, class_num=class_num)

        # Save the model if the current accuracy is better than the best accuracy
        if test_acc > best_accuracy:
            torch.save(_network, os.path.join(dir_, "{}_best_n1_{}_{}.pth".format(ceil, _seed, _cur_task)))
            best_accuracy = test_acc
            wandb.log({"acc":test_class_accuracy, "prec": test_class_precision, "rec":test_class_recall})
            #metric.update(logits, targets)
            cm = metric(logits, targets)
            
            fig_, ax = plt.subplots(figsize=(15, 15))
            
            wandb.log({"metrics" : metrics(preds, targets)}) 
            sns.heatmap(cm.cpu(), annot=True, fmt=".2f", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            
            wandb.log({"Conf. Matrix":wandb.Image(fig_)})
           
        wandb.log({"train_acc": train_acc, "test_acc": test_acc, "loss": losses/len(train_loader), "best_acc": best_accuracy})
        info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}'.format(
            _cur_task, epoch + 1, run_epoch, losses / len(train_loader), train_acc, test_acc)
        prog_bar.set_description(info)
    # logging.info(info)
    
def _compute_accuracy_domain( model, loader ,device = _device):
    model.eval()
    correct, total = 0, 0
        
    for i, (_,inputs, targets, bool_, img_) in enumerate(loader):
        inputs = inputs.to(_device)
        targets = torch.tensor([int(t) for t in targets]).to(_device)
        
        with torch.no_grad():
            outputs = model(inputs)

        probabilities = F.softmax(outputs, dim=1)
        preds = torch.max(probabilities, dim=1)[1]
       
        correct += preds.eq(targets).sum().item()
        total += targets.size(0)
       
    return np.around(correct * 100 / total, decimals=2)

def _compute_accuracy_past_domain( model,current, past, past_loader, device = _device):
    model.eval()
    correct, total = 0, 0
#    metric = ConfusionMatrix(task = 'multiclass',num_classes=class_num,normalize = 'all').to(device)
    metrics = MetricCollection({f'{current}_{past}_micro_recall': Recall(task = 'multiclass',num_classes=class_num, average='micro'), f'{current}_{past}_macro_recall': Recall(task = 'multiclass',num_classes=class_num, average='macro'), f'{current}_{past}_micro_precision': Precision(task = 'multiclass',num_classes=class_num, average='micro'),f'{current}_{past}_macro_precision': Precision(task = 'multiclass',num_classes=class_num, average='macro')}).to(device)
    
    labels = ["Anthropized", "Natural","Vegetation", "Agricultural","Herbeceous"]
    # labels = [
    #     "building", "pervious surface","impervious surface",
    #     "bare soil","water","coniferous",  "deciduous",
    #     "brushwood", "vineyard", "herbaceous vegetation",
    #     "agricultural land", "plowed land", "other" ]
    for i, (_,inputs, targets, bool_, img_) in enumerate(past_loader):
        inputs = inputs.to(_device)
        targets = torch.tensor([int(t) for t in targets]).to(_device)
        
        with torch.no_grad():
            outputs = model(inputs)
        probabilities = F.softmax(outputs, dim=1)
        preds = torch.max(probabilities, dim=1)[1]
        #metric.update(preds, targets)
        #cm = metric(preds, targets)
        metrics_report = metrics(preds, targets)

        correct += preds.eq(targets).sum().item()
        total += targets.size(0)
        #wandb.log({f"{current} Conf. Matrix {past}":wandb.Image(fig_)})
        wandb.log(metrics_report)
    return np.around(correct * 100 / total, decimals=2)

def compute_accuracy_per_class(model, loader, class_num, device=_device):
    model.eval()
    metric = MulticlassConfusionMatrix(num_classes=class_num, normalize='all').to(device)
    #metric = ConfusionMatrix(task = 'multiclass',num_classes=class_num, normalize='all').to(device)
    class_correct = [0] * class_num
    class_total = [0] * class_num

    # flair1_custom_classnames = {
    #     0: "Anthropized",
    #     1: "Natural",
    #     2: "Vegetation",
    #     3: "Agricultural",
    #     4: "Herbeceous"
    # }
    flair1_custom_classnames = {
        0 : "building",
        1 : "pervious surface",
        2 : "impervious surface",
        3 : "bare soil",
        4 : "water",
        5 : "coniferous",
        6 : "deciduous",
        7 : "brushwood",
        8 : "vineyard",
        9 : "herbaceous vegetation",
        10 : "agricultural land",
        11 : "plowed land",
        12 : "other"
        }

    for i, (_, inputs, targets, bool_, img_) in enumerate(loader):
        inputs = inputs.to(device)
        targets = torch.tensor([int(t) for t in targets]).to(device)
        with torch.no_grad():
            outputs = model(inputs)
        probabilities = F.softmax(outputs, dim=1)

        predicts = torch.max(probabilities, dim=1)[1]
        
        cm = metric(predicts, targets)

        for j in range(len(targets)):
            class_id = int(targets[j])
            class_total[class_id] += 1
            if predicts[j] == class_id:
                class_correct[class_id] += 1

    class_accuracy = {}
    class_precision = {}
    class_recall = {}

    confusion_matrix = cm.cpu().numpy()

    for i in range(class_num):
        if class_total[i] != 0:
            class_accuracy[f'{flair1_custom_classnames[i]}'] = np.around(class_correct[i] * 100 / class_total[i], decimals=2)

            true_positives = confusion_matrix[i, i]
            false_positives = confusion_matrix[:, i].sum() - true_positives
            false_negatives = confusion_matrix[i, :].sum() - true_positives

            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) != 0 else 0.0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) != 0 else 0.0

            class_precision[f'{flair1_custom_classnames[i]}'] = np.around(precision * 100, decimals=2)
            class_recall[f'{flair1_custom_classnames[i]}'] = np.around(recall * 100, decimals=2)
        else:
            class_accuracy[f'{flair1_custom_classnames[i]}'] = 0.0
            class_precision[f'{flair1_custom_classnames[i]}'] = 0.0
            class_recall[f'{flair1_custom_classnames[i]}'] = 0.0

    return class_accuracy, class_precision, class_recall

def main():
    
    random.seed(_seed)
    torch.manual_seed(_seed)
    torch.cuda.manual_seed(_seed)
    str_seed = str(_seed)
    print(f"Seed: {str_seed}")
    torch.autograd.set_detect_anomaly(True) 
    _network = ViTNet(_args).to(_device)
    past_accuracy_list = {}
    past_class_accuracy_list = {}
    past_recall_list= {}
    past_class_recall_list = {}
    past_class_precision_list = {}
    past_precision_list = {}
    bwt_list = {}
    sup_accuracy_list = {}
    test_loader_list = []
    ceil = 90
    seed = 77
    for _cur_task, task in enumerate(tasks_names):        
        
        
        train_paths = f'/scratcht/FLAIR_1/flair_txt_files/niveau1/{task}_train1_{seed}_{ceil}_.txt'
        test_paths = f'/scratcht/FLAIR_1/flair_txt_files/niveau1/{task}_test1_{seed}_{ceil}_.txt'
        
        train_dataset = DummyDataset(txt_file=train_paths, trsf=train_trsf)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                       num_workers=num_workers)
        
        test_dataset = DummyDataset(txt_file=test_paths, trsf=test_trsf)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                       num_workers=num_workers)
        test_loader_list.append(test_loader)
        wandb.init(
        project=f"INFN1-{ceil}-INP-{seed}-Replay",
        config={
        
        "_cur_task_": _cur_task,
        "task_name": task,
        "learning_rate": lrate,
        "architecture": "ViT-INP",
        "dataset": "FLAIR#1",
        "epochs": epochs,
        "topk" : topk, 
        "epochs" : epochs,
        "num_class" : 5, 
        "batch_size" : batch_size}, 
        name = f"{task}_{_cur_task}_seuil_{ceil}")
        task_name = tasks_names[_cur_task]
        past_accuracy_list[task_name] = {}
        past_class_accuracy_list[task_name] = {}
        past_recall_list[task_name] = {}
        past_precision_list[task_name] = {}
        past_class_precision_list[task_name] = {}
        past_class_recall_list[task_name]= {}
        # Définition du _network
        class_num = _network.class_num
        for name, param in _network.named_parameters():
            param.requires_grad_(False)
            if "classifier" in name:
                param.requires_grad_(True)
        # sup_model = torch.load( "sup_logs/best_n3_{}_{}.pth".format(seed,ceil)).to(_device)
        # if _cur_task==0:
        optimizer = optim.SGD(_network.parameters(), momentum=0.9,lr=init_lr,weight_decay=init_weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,T_max=init_epoch)
        run_epoch = init_epoch
        train_function(_network,train_loader, test_loader,optimizer,scheduler, run_epoch, _cur_task, seed = seed, ceil = ceil, class_num = class_num)
        model = torch.load(os.path.join(dir_, "{}_best_n3_{}_{}.pth".format(ceil, _seed, _cur_task))).to(_device)
        best_acc_cur_task = _compute_accuracy_domain(model, test_loader_list[_cur_task])
        best_class_acc_cur_task, best_prec_cur_task, best_rec_cur_task = compute_accuracy_per_class(model, test_loader_list[_cur_task], class_num)
        past_accuracy_list[task_name][task_name] = best_acc_cur_task
        past_class_accuracy_list[task_name][task_name] = best_class_acc_cur_task
        past_class_recall_list[task_name][task_name] = best_rec_cur_task
        past_class_precision_list[task_name][task_name] = best_prec_cur_task
        if _cur_task!=0 :
            for past in range(_cur_task):
                past_class_acc_cur_task, past_prec_cur_task, past_rec_cur_task = compute_accuracy_per_class(model, test_loader_list[past], class_num)
                past_accuracy_list[task_name][tasks_names[past]] = _compute_accuracy_past_domain(model= model,current = _cur_task, past = past, past_loader = test_loader_list[past])
                past_class_accuracy_list[task_name][tasks_names[past]] = past_class_acc_cur_task
                past_class_recall_list[task_name][tasks_names[past]] = past_rec_cur_task
                past_class_precision_list[task_name][tasks_names[past]] = past_prec_cur_task
            bwt = 0
            for task in range(_cur_task):
                bwt += (past_accuracy_list[task_name][tasks_names[task]] - past_accuracy_list[tasks_names[task]][tasks_names[task]])
            bwt_list[task_name] = bwt/_cur_task
        df = pd.DataFrame.from_dict(past_accuracy_list, orient='index')
        class_accuracy_df = pd.DataFrame.from_dict(past_class_accuracy_list, orient='index')
        class_recall_df =  pd.DataFrame.from_dict(past_class_recall_list, orient='index')
        class_prec_df = pd.DataFrame.from_dict(past_class_precision_list, orient='index')
        
        perf_df = wandb.Table(dataframe=df)
        class_acc_df = wandb.Table(dataframe=class_accuracy_df)
        class_rec_df = wandb.Table(dataframe=class_recall_df)
        class_prec_df = wandb.Table(dataframe=class_prec_df)
        
        bwt_df =  pd.DataFrame.from_dict(bwt_list, orient='index')
        bwt_df = wandb.Table(dataframe=bwt_df)
        wandb.log({"Accuracy": perf_df}) 
        wandb.log({"Class_Accuracy": class_acc_df})
        wandb.log({"Class_Precision": class_prec_df}) 
        wandb.log({"Class_Recall": class_rec_df}) 
        wandb.log({"BWT" : bwt_df})
        
        
        wandb.finish()
        _network = model
    
if __name__ == '__main__':
    main()
