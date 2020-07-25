#! /usr/bin/env python3

import numpy as np
from torch.utils.data import Dataset
from torchvision import datasets
import gdown
from gdown.cached_download import assert_md5sum
import os
import tarfile
from ..utils import common_functions as c_f

class CUB200(Dataset):
    url = 'https://drive.google.com/uc?id=1hbzc_P1FuxMkcabkgn9ZKinBwW683j45'
    filename = 'CUB_200_2011.tgz'
    md5 = '97eceeb196236b17998738112f37df78'

    def __init__(self, root, transform=None, download=False):
        self.root = os.path.join(root, "cub2011")
        if download:
            self.download_dataset()
        img_folder = os.path.join(self.root, "CUB_200_2011", "images")
        self.dataset = datasets.ImageFolder(img_folder)
        self.labels = np.array([b for (a, b) in self.dataset.imgs])
        self.transform = transform
        assert len(np.unique(self.labels)) == 200
        assert self.__len__() == 11788

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        if self.transform is not None:
            img = self.transform(img)
        output_dict = {"data": img, "label": label}
        return output_dict

    def download_dataset(self):
        c_f.makedir_if_not_there(self.root)
        output_location = os.path.join(self.root, self.filename)
        gdown.download(self.url, output_location, quiet=False)
        assert_md5sum(output_location, self.md5)
        with tarfile.open(output_location, "r:gz") as tar:
            tar.extractall(path=self.root, members = c_f.extract_progress(tar))
