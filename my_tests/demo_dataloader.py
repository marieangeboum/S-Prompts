import os
import numpy as np
from PIL import Image
import rasterio
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms

def rasterio_loader(path):
    
    with rasterio.open(path, 'r') as f:
        image_rasterio = f.read((1,2,3), out_dtype=np.uint8)
    # Convert the NumPy array to a PIL image
   
    image_rasterio_transposed = np.transpose(image_rasterio, (1, 2, 0))
  
    image = Image.fromarray(image_rasterio_transposed) 
    
    # to_tensor = transforms.ToTensor()
    # image = to_tensor(image)
    return image

class DummyDataset(Dataset):
    # Custom dataset for the data loader
    def __init__(self, txt_file, trsf):
        with open(txt_file, 'r') as f:
            lines = f.read().splitlines()

        self.data = []
        for line in lines:
            path, label = line.split(" ")
            self.data.append((path, label))

        self.trsf = trsf

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path, label = self.data[idx]

        image = self.trsf(rasterio_loader('/run/user/108646/gvfs/sftp:host=baymax/'+path))

        return idx, image, label


# Path to the .txt file containing the paths of images and labels
txt_file_path = f'/run/user/108646/gvfs/sftp:host=baymax/scratcht/T1_label_1/D004_2021_test.txt'

# Define any transformations you want to apply to your data
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# Create custom dataset
custom_dataset = DummyDataset(txt_file=txt_file_path, trsf=transform)

# Create a data loader
batch_size = 4
data_loader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=True)

# Iterate over the data loader
for batch in data_loader:
    indexes, images, labels = batch
    # Your training or evaluation code here
    print(f'Indexes: {indexes}')
    print(f'Labels: {labels}')
    print(f'Images Shape: {images.shape}')
    break  # Break after one iteration for demo purposes

