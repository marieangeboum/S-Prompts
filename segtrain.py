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