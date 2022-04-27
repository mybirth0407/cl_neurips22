import numpy as np
import torch
from PIL import Image
import cv2
import time

import random
from PIL import ImageFilter
from torchvision import transforms
from torch.utils.data import DataLoader

from mnist import ColourBiasedMNIST_BG, ColourBiasedMNIST_FG
# from torchvision.datasets import CIFAR10

def get_data_loader(batch_size, mode='BG', train=True, transform=None,
                    data_label_correlation=1.0, data_indices=None, 
                    colormap_idxs=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]):
    """
    mode: 'FG'(foreground) or 'BG'(background)
    """
    if mode == 'FG':
        Colored_MNIST = ColourBiasedMNIST_FG
    elif mode == 'BG':
        Colored_MNIST = ColourBiasedMNIST_BG
    else:
        raise NotImplemented

    dataset = Colored_MNIST(root='./', train=train, transform=transform,
                            download=True, data_label_correlation=data_label_correlation, 
                            data_indices=data_indices, colormap_idxs=colormap_idxs)
    dataloader = DataLoader(dataset=dataset,
                    batch_size=batch_size,
                    shuffle=train,
                    num_workers=4,
                    pin_memory=True)
    return dataloader

class ContraAugTransform():
    def __init__(self, aug_transform, base_transform) -> None:
        self.aug_transform = aug_transform
        self.base_transform = base_transform

    def __call__(self, x):
        t1 = self.aug_transform(x)
        t2 = self.aug_transform(x)
        x = self.base_transform(x)

        return [t1, t2, x]

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5),
                        std=(0.5, 0.5, 0.5))])

transform_aug = transforms.Compose([
            transforms.RandomResizedCrop(28, scale=(0.8, 1.)),
            # transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            # transforms.RandomHorizontalFlip(),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5),
                            std=(0.5, 0.5, 0.5))
        ])

# class cifar10(CIFAR10):
#     def __init__(
#         self,
#         root,
#         classes=range(10),
#         train=True,
#         transform=None,
#         target_transform=None,
#         download=False,
#         mean_image=None,
#     ):
#         super(cifar10, self).__init__(
#             root,
#             train=train,
#             transform=transform,
#             target_transform=target_transform,
#             download=download,
#         )

#         self.tensorTranform = transforms.ToTensor()
#         self.train = train
#         self.img_size = 32
#         if mean_image is not None:
#             mean_image = mean_image.transpose(1, 2, 0)
#             self.mean_image = cv2.resize(mean_image, (self.img_size, self.img_size))
#             self.mean_image = self.mean_image.transpose(2, 0, 1)

#         # Select subset of classes
#         if self.train:
#             self.train_data = self.data
#             self.train_labels = self.targets
#             train_data = []
#             train_labels = []

#             for i in range(len(self.train_data)):
#                 if self.train_labels[i] in classes:
#                     curr_img = cv2.resize(
#                         self.train_data[i], (self.img_size, self.img_size)
#                     )
#                     curr_img = curr_img.transpose(2, 0, 1)
#                     if mean_image is None:
#                         train_data.append(curr_img / 255.0)
#                     else:
#                         train_data.append(curr_img / 255.0 - self.mean_image)

#                     train_labels.append(int(self.train_labels[i]))

#             self.train_data = np.array(train_data, dtype=np.float32)
#             self.train_labels = np.array(train_labels)

#             if mean_image is None:
#                 self.mean_image = np.mean(self.train_data, axis=0)

#         else:
#             self.test_data = self.data
#             self.test_labels = self.targets
#             test_data = []
#             test_labels = []

#             for i in range(len(self.test_data)):
#                 if self.test_labels[i] in classes:
#                     curr_img = cv2.resize(
#                         self.test_data[i], (self.img_size, self.img_size)
#                     )
#                     curr_img = curr_img.transpose(2, 0, 1)
#                     test_data.append(curr_img / 255.0 - self.mean_image)
#                     test_labels.append(int(self.test_labels[i]))

#             self.test_data = np.array(test_data, dtype=np.float32)
#             self.test_labels = test_labels

#     def __getitem__(self, index):
#         if self.train:
#             image = self.train_data[index]
#             random_cropped = np.zeros(image.shape, dtype=np.float32)
#             padded = np.pad(image, ((0, 0), (4, 4), (4, 4)), mode="constant")
#             crops = np.random.random_integers(0, high=8, size=(1, 2))
#             # Cropping and possible flipping
#             if np.random.randint(2) > 0:
#                 random_cropped[:, :, :] = padded[
#                     :,
#                     crops[0, 0] : (crops[0, 0] + self.img_size),
#                     crops[0, 1] : (crops[0, 1] + self.img_size),
#                 ]
#             else:
#                 random_cropped[:, :, :] = padded[
#                     :,
#                     crops[0, 0] : (crops[0, 0] + self.img_size),
#                     crops[0, 1] : (crops[0, 1] + self.img_size),
#                 ][:, :, ::-1]
#             image = torch.FloatTensor(random_cropped)
#             target = self.train_labels[index]
#         else:
#             image, target = self.test_data[index], self.test_labels[index]

#         image = torch.FloatTensor(image)

#         return index, image, target

#     def __len__(self):
#         if self.train:
#             return len(self.train_data)
#         else:
#             return len(self.test_data)


# class cifar100(cifar10):
#     base_folder = "cifar-100-python"
#     url = "http://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
#     filename = "cifar-100-python.tar.gz"
#     tgz_md5 = "eb9058c3a382ffc7106e4002c42a8d85"
#     train_list = [
#         ["train", "16019d7e3df5f24257cddd939b257f8d"],
#     ]
#     test_list = [
#         ["test", "f0ef6b0ae62326f3e7ffdfab6717acfc"],
#     ]
#     meta = {
#         "filename": "meta",
#         "key": "fine_label_names",
#         "md5": "7973b15100ade9c7d40fb424638fde48",
#     }
