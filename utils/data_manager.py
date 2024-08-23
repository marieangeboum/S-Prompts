import os
import torch
import logging
import rasterio
import imageio
import numpy as np
from PIL import Image

from itertools import product
from torch.utils.data import DataLoader, RandomSampler, ConcatDataset, Dataset
from torchvision import transforms
from utils.data import  iFlair1
from rasterio.windows import Window

def _get_idata(dataset_name, args=None):
    name = dataset_name.lower()
    if name == 'flair':
        return iFlair1(args)
    else:
        raise NotImplementedError('Unknown dataset {}.'.format(dataset_name))
        
        
class DataManager(object):
    def __init__(self, dataset_name, shuffle, seed, init_cls, increment,args=None):
        self.args = args
        self.dataset_name = dataset_name # nom du jeu de données
        
        self._setup_data(dataset_name, shuffle, seed)
        
        assert init_cls <= len(self._class_order), 'Not enough classes.' # initialisation du nombre de classes
        self._increments = [init_cls]# Nombre de tâche incrémentales  c'est égale au nombre de classe  si on est dans le CIL
        while sum(self._increments) + increment < len(self._class_order):
            self._increments.append(increment)
        offset = len(self._class_order) - sum(self._increments)
        if offset > 0:
            self._increments.append(offset)

    @property
    def nb_tasks(self):
        "Returns the nb of tasks"
        return len(self._increments)
    
    def _setup_data(self, dataset_name, shuffle, seed):
        idata = _get_idata(dataset_name, self.args)
        # idata.download_data()
        idata.download_data_segmentation()

        # Data
        # self._train_data, self._train_targets = idata.train_data, idata.train_targets
        # self._test_data, self._test_targets = idata.test_data, idata.
        self._train_data, self._train_targets = idata.train_data_paths, idata.train_target_paths
        self._test_data, self._test_targets = idata.test_data_paths, idata.test_target_paths
        self._val_data, self._val_targets = idata.val_data_paths, idata.val_target_paths

        self.use_path = idata.use_path
        
        # Transforms ( redéfinir )
        self._train_trsf = idata.train_trsf
        self._test_trsf = idata.test_trsf
        self._common_trsf = idata.common_trsf

        # Order
        order = [i for i in range(len(np.unique(self._train_targets)))]
        if shuffle:
            np.random.seed(seed)
            order = np.random.permutation(len(order)).tolist()
        else:
            order = idata.class_order
        self._class_order = order
        logging.info(self._class_order)

        # Map indices
        # self._train_targets = _map_new_class_index(self._train_targets, self._class_order)
        # self._test_targets = _map_new_class_index(self._test_targets, self._class_order)

    def get_task_size(self, task):
        return self._increments[task]

    def get_dataset(self, indices, source, mode, appendent=None, ret_data=False):
        # if source == 'train':
        #     x, y = self._train_data, self._train_targets
        # elif source == 'test':
        #     x, y = self._test_data, self._test_targets
        # else:
        #     raise ValueError('Unknown data source {}.'.format(source))
        if source == 'train':
            x, y = self._train_data, self._train_targets
            fixed_crop = False
        elif source == 'val':
            x, y = self._test_data, self._test_targets
            fixed_crop = True
        elif source == 'test':
            x, y = self._test_data, self._test_targets
            fixed_crop = True
        else:
            raise ValueError('Unknown data source {}.'.format(source))

        if mode == 'train':
            trsf = transforms.Compose([*self._train_trsf, *self._common_trsf])
        elif mode == 'flip':
            trsf = transforms.Compose([*self._test_trsf, transforms.RandomHorizontalFlip(p=1.), *self._common_trsf])
        elif mode == 'test' or mode == 'val':
            trsf = transforms.Compose([*self._test_trsf, *self._common_trsf])
        else:
            raise ValueError('Unknown mode {}.'.format(mode))

        data, targets = [], []
        # for idx in indices:
        #     class_data, class_targets = self._select(x, y, low_range=idx, high_range=idx+1)
        #     data.append(class_data)
        #     targets.append(class_targets)
        #
        # if appendent is not None and len(appendent) != 0:
        #     appendent_data, appendent_targets = appendent
        #     data.append(appendent_data)
        #     targets.append(appendent_targets)
        #
        # data, targets = np.concatenate(data), np.concatenate(targets)

        for item1, item2 in zip(x, y):
            data.append(FlairDs(image_path=item1[0], label_path=item2[0], task_id=item1[1],
                                   tile = Window(col_off=0, row_off=0, width=512, height=512),
                                   fixed_crops = fixed_crop,
                                   crop_size = 224,
                                   crop_step = 224))

        dataset = ConcatDataset(data)
        if ret_data: 
            return data, dataset
        else:
            return dataset

    def get_anchor_dataset(self, mode, appendent=None, ret_data=False):
        if mode == 'train':
            trsf = transforms.Compose([*self._train_trsf, *self._common_trsf])
        elif mode == 'flip':
            trsf = transforms.Compose([*self._test_trsf, transforms.RandomHorizontalFlip(p=1.), *self._common_trsf])
        elif mode == 'test':
            trsf = transforms.Compose([*self._test_trsf, *self._common_trsf])
        else:
            raise ValueError('Unknown mode {}.'.format(mode))

        data, targets = [], []
        if appendent is not None and len(appendent) != 0:
            appendent_data, appendent_targets = appendent
            data.append(appendent_data)
            targets.append(appendent_targets)

        data, targets = np.concatenate(data), np.concatenate(targets)

        if ret_data:
            return data, targets, DummyDataset(data, targets, trsf, self.use_path)
        else:
            return DummyDataset(data, targets, trsf, self.use_path)

    def get_dataset_with_split(self, indices, source, mode, appendent=None, val_samples_per_class=0):
        if source == 'train':
            x, y = self._train_data, self._train_targets
        elif source == 'test':
            x, y = self._test_data, self._test_targets
        else:
            raise ValueError('Unknown data source {}.'.format(source))

        if mode == 'train':
            trsf = transforms.Compose([*self._train_trsf, *self._common_trsf])
        elif mode == 'test':
            trsf = transforms.Compose([*self._test_trsf, *self._common_trsf])
        else:
            raise ValueError('Unknown mode {}.'.format(mode))

        train_data, train_targets = [], []
        val_data, val_targets = [], []
        for idx in indices:
            class_data, class_targets = self._select(x, y, low_range=idx, high_range=idx+1)
            val_indx = np.random.choice(len(class_data), val_samples_per_class, replace=False)
            train_indx = list(set(np.arange(len(class_data))) - set(val_indx))
            val_data.append(class_data[val_indx])
            val_targets.append(class_targets[val_indx])
            train_data.append(class_data[train_indx])
            train_targets.append(class_targets[train_indx])

        if appendent is not None:
            appendent_data, appendent_targets = appendent
            for idx in range(0, int(np.max(appendent_targets))+1):
                append_data, append_targets = self._select(appendent_data, appendent_targets,
                                                           low_range=idx, high_range=idx+1)
                val_indx = np.random.choice(len(append_data), val_samples_per_class, replace=False)
                train_indx = list(set(np.arange(len(append_data))) - set(val_indx))
                val_data.append(append_data[val_indx])
                val_targets.append(append_targets[val_indx])
                train_data.append(append_data[train_indx])
                train_targets.append(append_targets[train_indx])

        train_data, train_targets = np.concatenate(train_data), np.concatenate(train_targets)
        val_data, val_targets = np.concatenate(val_data), np.concatenate(val_targets)

        return DummyDataset(train_data, train_targets, trsf, self.use_path), \
            DummyDataset(val_data, val_targets, trsf, self.use_path)

    def _select(self, x, y, low_range, high_range):
        idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]
        return x[idxes], y[idxes]

class DummyDataset(Dataset): # custom dataset pour le dataloader 
    def __init__(self, images, labels, trsf, use_path=False):
        assert len(images) == len(labels), 'Data size error!'
        self.images = images
        self.labels = labels
        self.trsf = trsf
        self.use_path = use_path

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if self.use_path:
            image = self.trsf(rasterio_loader(self.images[idx])) 
        else:
            image = self.trsf(Image.fromarray(self.images[idx]))
        label = self.labels[idx]
        return idx, image, label

def _map_new_class_index(y, order):
    return np.array(list(map(lambda x: order.index(x), y)))

def pil_loader(path):
    '''
    Ref:
    https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#ImageFolder
    '''
    with open(path, 'rb') as f: # equivalent de f.read de rasterio
       img = Image.open(f)
       return img.convert('RGB')

def rasterio_loader(path):
    
    with rasterio.open(path, 'r') as f:
        image_rasterio = f.read((1,2,3), out_dtype=np.uint8)
    # Convert the NumPy array to a PIL image
   
    image_rasterio_transposed = np.transpose(image_rasterio, (1, 2, 0))
  
    image = Image.fromarray(image_rasterio_transposed) 
    
    to_tensor = transforms.ToTensor()
    image = to_tensor(image)
    return image#.convert('RGB')

def accimage_loader(path):
    '''
    Ref:
    https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#ImageFolder
    accimage is an accelerated Image loader and preprocessor leveraging Intel IPP.
    accimage is available on conda-forge.
    '''
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    '''
    Ref:
    https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#ImageFolder
    '''
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)

def get_tiles(nols, nrows, size, size2=None, step=None, step2=None, col_offset=0, row_offset=0):
    
    if step is None: step = size
    if size2 is None: size2 = size
    if step2 is None: step2 = step

    max_col_offset = int(np.ceil((nols-size)/step))
    # Remove all offsets such that offset+size > nols and add one offset to
    # reach nols
    col_offsets = list(range(col_offset, col_offset + nols, step))[:max_col_offset+1]
    col_offsets[max_col_offset] = col_offset + nols - size

    max_row_offset = int(np.ceil((nrows-size2)/step2))
    # Remove all offsets such that offset+size > nols and add one offset to
    # reach nols
    row_offsets = list(range(row_offset, row_offset + nrows, step2))[:max_row_offset+1]
    row_offsets[max_row_offset] = row_offset + nrows - size2

    offsets = product(col_offsets, row_offsets)
    big_window = Window(col_off=col_offset, row_off=row_offset, width=nols, height=nrows)
    for col_off, row_off in offsets:
        window = Window(col_off=col_off, row_off=row_off, width=size,
                        height=size2).intersection(big_window)
        yield window


class FlairDs(Dataset):
    def __init__(self,image_path,tile,fixed_crops, crop_size,crop_step, task_id,img_aug= None,
                 label_path=None, *args,**kwargs):

        self.image_path = image_path # path to image
        self.label_path = label_path # pth to corresponding label
        self.tile = tile # initializing a tile to be extracted from image
        self.crop_windows = list(get_tiles(
            nols=tile.width,
            nrows=tile.height, 
            size=crop_size,
            step=crop_step,
            row_offset=tile.row_off,
            col_offset=tile.col_off)) if fixed_crops else None # returns a list of tile these are crop extracted from the initial img
        self.crop_size = crop_size,# initializing crop size
        self.img_aug= img_aug,
        self.task_id = task_id
    def read_label(self, label_path, window):
        pass

    def read_image(self, image_path, window):
        pass

    def __len__(self):
        # returns nb of cropped windows
        return len(self.crop_windows) if self.crop_windows else 1

    def __getitem__(self, idx):
        ''' Given index idx this function loads a sample  from the dataset based on index.
            It identifies image'location on disk, converts it to a tensor
        '''
        if self.crop_windows:# if self.windows is initialized correctly begin iteration on said list
            window = self.crop_windows[idx]
        else: # otherwise use Winodw method from rasterio module to initilize a window of size cx and cy
            cx = self.tile.col_off + np.random.randint(0, self.tile.width - self.crop_size + 1) # why add those randint ?
            cy = self.tile.row_off + np.random.randint(0, self.tile.height - self.crop_size + 1)
            window = Window(cx, cy, self.crop_size, self.crop_size)
        ## Not here --> # vizualise the window crops extracted from the input image
        with rasterio.open(self.image_path, 'r') as image_file:
            image_rasterio = image_file.read(window=window, out_dtype=np.float32) # read the cropped part of the image
            img_path_strings = self.image_path.split('/')
            domain_pattern = img_path_strings[-4]

        # converts image crop into a tensor more precisely returns a contiguous tensor containing the same data as self tensor.
        image = torch.from_numpy(image_rasterio).float().contiguous()
        label = None
        if self.label_path:

            with rasterio.open(self.label_path, 'r') as label_file:
                label = label_file.read(window=window, out_dtype=np.float32)

            # converts label crop into contiguous tensor
            label = torch.from_numpy(label).float().contiguous()
            mask = label >= 13
            label[mask] = 13
            multi_labels = label.float()
            multi_labels -= 1
            final_label = multi_labels
            final_label = final_label * (self.task_id + 1)

        if self.img_aug is not None:            
            final_image, final_mask = self.img_aug(img=image, label=final_label)
        else:
            final_image, final_mask = image, final_label
        return {'orig_image':image,
                'orig_mask': label,
                'id' : domain_pattern,
                'window':window,
                'image':final_image,
                'mask':final_mask}
