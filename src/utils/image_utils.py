import torch
import torchvision.transforms as transforms
import numpy as np

STD_IMAGENET = torch.tensor([0.2023, 0.1994, 0.2010]).view(1, 1, 3)
MEAN_IMAGENET = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 1, 3)
STD_PATHMNIST = torch.tensor([0.5, 0.5, 0.5]).view(1, 1, 3)
MEAN_PATHMNIST = torch.tensor([0.5, 0.5, 0.5]).view(1, 1, 3)

def train_prep_cifar():
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.Pad(4),
        transforms.RandomCrop(32),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
    ])

def valid_prep_cifar():
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
    ])

def train_prep_fmnist():
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

def valid_prep_fmnist():
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
    ])

def train_prep_pathmnist():
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN_PATHMNIST.numpy(), std=STD_PATHMNIST.numpy()),
    ])

def valid_prep_pathmnist():
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN_PATHMNIST.numpy(), std=STD_PATHMNIST.numpy()),
    ])

def train_prep_eurosat():
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.Pad(4),
        transforms.RandomCrop(64),
        transforms.ToTensor(),
    ])

def valid_prep_eurosat():
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
    ])

def cutout(images, mask_size):
    _, h, w = images.shape
    y = np.random.randint(h)
    x = np.random.randint(w)
    y1 = np.clip(y - mask_size // 2, 0, h)
    y2 = np.clip(y + mask_size // 2, 0, h)
    x1 = np.clip(x - mask_size // 2, 0, w)
    x2 = np.clip(x + mask_size // 2, 0, w)
    images[:, y1:y2, x1:x2] = 0
    return images

def cutout_cifar(images):
    images = cutout(images, 16)
    return images

def cutout_fmnist(images):
    images = cutout(images, 14)
    return images

def cutout_pathmnist(images):
    images = cutout(images, 14)
    return images

def cutout_eurosat(images):
    images = cutout(images, 32)
    return images
