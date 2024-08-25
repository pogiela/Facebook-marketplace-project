'''
This module contains the class to create a custom dataset for the images

Classes:
    CustomImageDataset: Custom dataset class for the images
        methods: __init__, __len__, __getitem__
'''

import os
import sys
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image


class CustomImageDataset(Dataset):
    '''
    Custom dataset class for the images
    
    Args:
        datasource_file (string): Path to the CSV file with annotations
        img_dir (string): Directory with all the images
        transform (callable, optional): Optional transformations to be applied to an image
    '''   
    def __init__(self, datasource_file, img_dir, transform=None):
        self.img_labels = pd.read_csv(datasource_file)
        self.img_dir = img_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 1] + '.jpg')
        image = read_image(img_path)
        # image = read_image(img_path).float() / 255.0  # Convert images to floating-point tensors and normalize to [0, 1]
        
        # Check the number of channels and convert grayscale to RGB
        if image.shape[0] == 1:  # Grayscale image
            image = image.repeat(3, 1, 1)  # Repeat the single channel 3 times

        label = self.img_labels.iloc[idx, 3]
        
        if self.transform:
            image = self.transform(image)
            
        return (image, torch.tensor(label))
    

if __name__ == "__main__":
    print('This is a module containing functions to load and clean tabular data')
    print('Please run the main programme to execute the data processing')
    sys.exit()