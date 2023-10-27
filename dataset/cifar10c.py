import numpy as np
import os

from PIL import Image
from torchvision import datasets as dset

from .utils import load_txt

class CIFAR10C(dset.VisionDataset):
    def __init__(self, root :str, cname :str,
                 transform=None, target_transform=None, 
                 severity=0):
        super(CIFAR10C, self).__init__(
            root, transform=transform,
            target_transform=target_transform
        )
        self.CORRUPTIONS = load_txt(os.path.join(root,'corruptions.txt'))
        # self.mean = (0.4914, 0.4822, 0.4465)
        # self.std = (0.2471, 0.2435, 0.2616)        
        self.mean = (0, 0, 0)
        self.std = (1, 1, 1)
        
        self.n_classes = 10

        data_path = os.path.join(root, cname + '.npy')
        target_path = os.path.join(root, 'labels.npy')
        
        self.data = np.load(data_path)
        self.targets = np.load(target_path)        
        if 0 <= severity <=5:
            if severity > 0:
                self.data = self.data[(severity * 10000)-10000:severity * 10000]
                self.targets = self.targets[(severity * 10000)-10000:severity * 10000]
        else:
            raise ValueError("Severity level not supported")

    def __getitem__(self, index):
        img, targets = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            targets = self.target_transform(targets)

        return img, targets, index
    
    def __len__(self):
        return len(self.data)