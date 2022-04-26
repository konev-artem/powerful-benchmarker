from pytorch_metric_learning.trainers.base_trainer import BaseTrainer
from pytorch_metric_learning.trainers.metric_loss_only import MetricLossOnly
import torch 

import torchvision.transforms as transforms
from PIL import Image, ImageOps, ImageFilter
import random

class GaussianBlur(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            sigma = random.random() * 1.9 + 0.1
            return img.filter(ImageFilter.GaussianBlur(sigma))
        else:
            return img


class Solarization(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img

class Transform:
    def __init__(self):
        self.toPIL = transforms.ToPILImage()
        self.toTensor = transforms.ToTensor()

        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(224, interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                        saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(p=1.0),
            Solarization(p=0.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.transform_prime = transforms.Compose([
            transforms.RandomResizedCrop(224, interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                        saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(p=0.1),
            Solarization(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __call__(self, x):
        x = self.toPIL(x)
        y1 = self.transform(x)
        y2 = self.transform_prime(x)
        return y1, y2
        

class BarlowTwinTrainer(MetricLossOnly):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.barlow_twin_transfrom = Transform()
        self.collate_fn = self.custom_collate_fn
        self.initialize_dataloader()

    def custom_collate_fn(self, data):
        transformed_data, labels = [], []
        for i, d in enumerate(data):

            img, img_label = self.data_and_label_getter(d)
            
            img1, img2 = self.barlow_twin_transfrom(img)

            transformed_data.append(img1)
            transformed_data.append(img2)

            labels.append(img_label)
            labels.append(img_label)

        return {'data': torch.stack(transformed_data, dim=0),
         'label': torch.LongTensor(labels)}

