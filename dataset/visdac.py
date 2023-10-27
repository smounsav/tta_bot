from typing import Optional, Callable, Tuple, Any, List
from torchvision import datasets as dset
import torchvision.transforms as T

import os
from torchvision.datasets.folder import default_loader

class VisDAC(dset.VisionDataset):
    def __init__(self, root :str,
                 transform=None, target_transform=None):
        super(VisDAC, self).__init__(
            root, transform=transform,
            target_transform=target_transform
        )
        self.transform = transform
        self.target_transform = target_transform
        self.loader = default_loader

        data_list_file = os.path.join(root, 'image_list.txt')

        self.dataset = self.parse_data_file(data_list_file)
        

        
        self.mean = (0, 0, 0)
        self.std = (1, 1, 1)
        
        self.n_classes = 12

    def parse_data_file(self, file_name: str) -> List[Tuple[str, int]]:
        """Parse file to data list

        Parameters:
            - **file_name** (str): The path of data file
            - **return** (list): List of (image path, class_index) tuples
        """
        with open(file_name, "r") as f:
            data_list = []
            for line in f.readlines():
                path, target = line.split()
                if not os.path.isabs(path):
                    path = os.path.join(self.root, path)
                target = int(target)
                data_list.append((path, target))
        return data_list

    def __getitem__(self, index):
        path, targets = self.dataset[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None and target is not None:
            target = self.target_transform(target)
                    
        img = T.functional.normalize(img, self.mean, self.std)
        
        return img, targets, index
    
    def __len__(self):
        return len(self.dataset)