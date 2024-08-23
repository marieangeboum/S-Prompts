import os
import glob
import random
import numpy as np
from torchvision import datasets, transforms
from utils.toolkit import split_images_labels
from utils.datautils.core50data import CORE50


class iData(object):
    train_trsf = []
    test_trsf = []
    common_trsf = []
    class_order = None

class iFlair1(iData):

    use_path = True
    train_trsf = [
        
        transforms.Resize(224, antialias=True),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=63/255)
    ]
    test_trsf = [
        # transforms.ToTensor(),
        transforms.Resize(256, antialias=True),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        # transforms.ToTensor(),
        transforms.Normalize(mean=[0,0,0], std=[1,1,1]),
    ]

    def __init__(self, args):
        self.args = args
        class_order = np.arange(args["init_cls"]*args["total_sessions"]).tolist() # j'appellerai pas ça class order... ce sont l'ensemble de classe par domaine elles ont un nom différent par domaine
        self.class_order = class_order
        self.domain_names = args["task_name"]


    def download_data(self):
        self.image_list_root = self.args["data_path"]
        image_list_paths = [os.path.join(self.image_list_root, d + "_" + "train" + ".txt") for d in self.domain_names]
        imgs = []
        print(image_list_paths)
        for taskid, image_list_path in enumerate(image_list_paths):
            image_list = open(image_list_path).readlines()
            imgs += [((val).rsplit(maxsplit=1)[0], int(val.rsplit(maxsplit=1)[-1]) + taskid *self.args["init_cls"]) for val in image_list] 
        train_x, train_y = [], []
        for item in imgs:
            train_x.append(os.path.join(self.image_list_root, item[0]))
            train_y.append(item[1])
        self.train_data = np.array(train_x)
        self.train_targets = np.array(train_y)

        image_list_paths = [os.path.join(self.image_list_root, d + "_" + "test" + ".txt") for d in self.domain_names]
        imgs = []
        for taskid, image_list_path in enumerate(image_list_paths):
            print(taskid, image_list_path)
            image_list = open(image_list_path).readlines()
            
            imgs += [((val).rsplit(maxsplit=1)[0], int(val.rsplit(maxsplit=1)[-1]) + taskid*self.args["init_cls"]) for val in image_list]
        test_x, test_y = [], []
        for item in imgs:
            test_x.append(os.path.join(self.image_list_root, item[0]))
            test_y.append(item[1])
        self.test_data = np.array(test_x)
        self.test_targets = np.array(test_y)

    def download_data_segmentation(self):
        image_list_paths = []
        self.image_list_root = self.args["data_path"]
        train_x, train_y = [], []
        val_x, val_y = [], []
        test_x, test_y = [], []
        image_list_paths.extend(glob.glob(os.path.join(self.image_list_root, '{}/Z*_*/img/IMG_*.tif'.format(domain))) for domain in self.domain_names)
        for taskid, domain in enumerate(self.domain_names):
            img = glob.glob(os.path.join(self.image_list_root, '{}/Z*_*/img/IMG_*.tif'.format(domain)))
            random.shuffle(img)
            train_imgs = [(item, taskid) for item in img[:int(len(img)*self.args["train_split_coef"])]]
            test_imgs = [(item, taskid) for item in img[int(len(img)*self.args["train_split_coef"]):]]
            val_imgs = [(item[0], item[1]) for item in train_imgs[:int(len(train_imgs)*self.args["train_split_coef"])]]
            train_masks = [(item[0].replace('img/IMG', 'msk/MSK'), item[1]) for item in train_imgs]
            test_masks = [(item[0].replace('img/IMG', 'msk/MSK'), item[1]) for item in test_imgs]
            val_masks = [(item[0].replace('img/IMG', 'msk/MSK'), item[1]) for item in val_imgs]

            random.shuffle(train_imgs)

            train_x.extend(train_imgs)
            train_y.extend(train_masks)

            val_x.extend(val_imgs)
            val_y.extend(val_masks)

            test_x.extend(test_imgs)
            test_y.extend(test_masks)
        self.train_data_paths = train_x
        self.train_target_paths = train_y
        self.val_data_paths = val_x
        self.val_target_paths = val_y
        self.test_data_paths = test_x
        self.test_target_paths = test_y
                                   




