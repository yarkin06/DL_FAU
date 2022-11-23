from torch.utils.data import Dataset
import torch
from pathlib import Path
from skimage.io import imread
from skimage.color import gray2rgb
import numpy as np
import torchvision as tv

train_mean = [0.59685254, 0.59685254, 0.59685254]
train_std = [0.16043035, 0.16043035, 0.16043035]


class ChallengeDataset(Dataset):
    # TODO implement the Dataset class according to the description
    def __init__(self, data, mode):
        self.data = data
        # self.train = (mode == 'train')
        self.mode = mode
        TF = tv.transforms
        self.val_transform = TF.Compose([  
                                    TF.ToPILImage(),
                                    TF.ToTensor(),
                                    TF.Normalize(train_mean, train_std),
                                    # TF.RandomHorizontalFlip(p=0.3),
                                    # TF.RandomVerticalFlip(p=0.3)
                                    ])
        self.train_transform = TF.Compose([  
                                    TF.ToPILImage(),
                                    TF.ToTensor(),
                                    TF.Normalize(train_mean, train_std)
                                    # TF.RandomHorizontalFlip(p=0.3),
                                    # TF.RandomVerticalFlip(p=0.3)
                                    ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.mode == "val":
            data = self.data.iloc[idx]
            img = imread(data['filename'], as_gray=True)
            img = gray2rgb(img)
            label = np.array([data['crack'], data['inactive']])
            img = self.val_transform(img)
            return img, label
        if self.mode == "train":
            data = self.data.iloc[idx]
            img = imread(data['filename'], as_gray=True)
            img = gray2rgb(img)
            label = np.array([data['crack'], data['inactive']])
            img = self.train_transform(img)
            return img, label

    @property
    def transform(self):
        return self._transform
    
    @transform.setter
    def transform(self, transform):
        self._transform = transform