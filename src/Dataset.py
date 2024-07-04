import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torch.nn as nn
import config
from PIL import Image
from glob import glob
import os
import numpy as np
from utils import *

#Slit画像のデータセット
class SlitDatasetBase(Dataset):
    def __init__(self,image_path_dir,transform=None):
        images,image_paths,labels,item_indexes = [],[],[],[]
        images_append = images.append
        self.image_path_dir = image_path_dir
        item_index = 0
        self.transform = transform

        image_Exfoliation_paths = glob(os.path.join(image_path_dir,"exfoliation_slit/*.jpg"))
        image_NORMAL_paths = glob(os.path.join(image_path_dir,"notexfoliation_slit/*.jpg"))

        # Exfoliation
        for image_path in image_Exfoliation_paths:
            item_indexes.append(item_index)
            item_index += 1
            image_paths.append(image_path)
            image = Image.open(image_path).convert('RGB')
            images_append(self.transform(image))
            labels.append(1)

        # NORMAL
        for image_path in image_NORMAL_paths:
            item_indexes.append(item_index)
            item_index += 1
            image_paths.append(image_path)
            image = Image.open(image_path).convert('RGB')
            images_append(self.transform(image))
            labels.append(0)

        self.image_paths = np.array(image_paths)
        self.images = images
        self.labels = np.array(labels)
        self.item_indexes = np.array(item_indexes)
        self.transform = transform


    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = self.images[index]
        label = torch.eye(config.n_class)[self.labels[index]]
        item_index = self.item_indexes[index]

        return (image,torch.Tensor(label),item_index) if self.image_path_dir==config.train_path else (image,torch.Tensor(label),image_path)

    def __len__(self):
        return len(self.image_paths)

    def pick_label(self, index):
        label = torch.eye(config.n_class)[self.labels[index]]
        return torch.Tensor(label)


class SlitDataset(SlitDatasetBase):
    def get_label(self, label_base):
        if label_base == 1:
            return 0
        else:
            return 1

def load_dataset():
    train_transform = \
        transforms.Compose([transforms.Resize(config.image_size),
                            transforms.CenterCrop(config.image_size),
                            transforms.ToTensor(),
                            transforms.Normalize((0.485,0.456,0.406),
                                                 (0.229,0.224,0.225))])
    test_transform = \
        transforms.Compose([transforms.Resize(config.image_size),
                            transforms.CenterCrop(config.image_size),
                            transforms.Grayscale(num_output_channels=3),
                            transforms.ToTensor(),
                            transforms.Normalize((0.485,0.456,0.406),
                                                 (0.229,0.224,0.225))])
    
    dataset = {}
    dataset['train'] = \
            SlitDataset(image_path_dir=config.train_path,
                                    transform=train_transform)

    dataset['test'] = \
            SlitDataset(image_path_dir=config.test_path,transform=test_transform)
    print('ok')

    return dataset