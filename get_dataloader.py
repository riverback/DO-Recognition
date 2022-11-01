import os
import random
import numpy as np
import cv2

import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F
from torch.utils.data import DataLoader, Dataset



def gaussian_noise(image: np.ndarray, mean: float, sigma: float):
    assert image.max() <= 1
    noise = np.random.normal(mean, sigma, image.shape)
    gaussian_image = image + noise
    gaussian_image = np.clip(gaussian_image, 0, 1)
    return gaussian_image

def id_collate(batch):
    from torch.utils.data.dataloader import default_collate
    new_batch = []
    ids = []
    for _batch in batch:
        new_batch.append(_batch[:-1])
        ids.append(_batch[-1])
    
    return default_collate(new_batch), ids

class OLF_Dataset(Dataset):
    
    MEAN, STD = 0., 1. # mean and std of the OLF train dataset
    
    def __init__(self, file_path, mode, aug_prob=0.5):
        self.file_path = file_path
        self.mode = mode
        self.aug_prob = aug_prob
        
        with open(file_path, 'r') as f:
            self.data = f.readlines()
            
        print("there are {} images in {} dataset\n".format(self.__len__(), mode))
        
    def __getitem__(self, index):
        image_path, label = self.data[index].strip('\n').split()
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) / 255
        # resize images to 56*56
        image = np.clip(cv2.resize(image, (56, 56)), 0, 1)
        
        label = int(label)
        if label == 0: # healthy slices have no masks
            image_id = image_path.strip('.png')[5:].replace('/', '-') # eg. Healthy_spines-005-0112
            mask = np.zeros(image.shape, dtype=int)
        else:
            image_id = image_path.strip('.png')[5:].replace('/', '-') #eg. OLF_spines-002-T11-OLF-00012
            mask = cv2.imread(image_path.strip('.png')+'-label.png')
            mask = cv2.resize(mask, (56,56))
            mask = np.where(mask>1, 1, 0)
        
        if self.mode == 'train' and random.random() <= self.aug_prob:
            image = gaussian_noise(image, 0., 1.)
    
        if len(image.shape) == 3:
            image = image[:,:,0]
        if len(mask.shape) == 3:
            mask = mask[:,:,0]
        
        image = np.expand_dims(image, 0)
        mask = np.expand_dims(mask, 0)
        
        image = torch.tensor(image).type(torch.FloatTensor)
        mask = torch.tensor(mask).type(torch.IntTensor)
        label = torch.tensor(label).long()

        Transforme = []
        # Transforme.append(T.Normalize(mean=self.MEAN, std=self.STD))
        Transforme = T.Compose(Transforme)
        
        image = Transforme(image)
        
        return image, mask, label, image_id
        
    def __len__(self):
        return len(self.data)


def get_loader(file_path, mode, aug_prob, batch_size, num_workers, dataset_type='OLF_Dataset'):
    if mode == 'train':
        shuffle = True
    else:
        shuffle = False
        
    if dataset_type == 'OLF_Dataset' or 'Spine_Dataset':
        data_loader = DataLoader(
            dataset=OLF_Dataset(file_path, mode, aug_prob),
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            collate_fn=id_collate)
    elif dataset_type == 'MIL_Dataset':
        # for Multi Instance Learning
        raise NotImplementedError()
    else:
        raise NotImplementedError()
    
    return data_loader
    