#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 10:07:56 2024

@author: becky
"""

import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from skimage.transform import resize, rotate
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
# import torchvision.transforms.functional as F
import torchvision.transforms.functional as TF

# class ToTensor(object):
#     """Convert ndarrays in sample to Tensors."""

#     def __call__(self, sample):
#         image, mask = sample['image'], sample['mask']

#         # swap color axis because
#         # numpy image: H x W x C
#         # torch image: C x H x W        
#         image = np.array(image)
#         image = image.transpose((2, 0, 1))
#         mask = np.array(mask)
#         mask = mask.transpose((2, 0, 1))
        
#         image = TF.normalize(torch.from_numpy(image), mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#         mask = TF.normalize(torch.from_numpy(mask), mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#         return {'image': torch.from_numpy(image),
#                 'mask': torch.from_numpy(mask)}

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W        
        image = np.array(image)
        image = image.transpose((2, 0, 1))
        mask = np.array(mask)
        mask = mask.transpose((2, 0, 1))

        # Convert numpy arrays to torch tensors and change to float
        image = torch.from_numpy(image).float()
        mask = torch.from_numpy(mask).float()

        # Normalize
        image = TF.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        mask = TF.normalize(mask, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        return {'image': image, 'mask': mask}

class Rotate(object):
    """Rotate an image and mask."""
    def __init__(self, angle):
        self.angle = angle

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']

        # Rotate image and mask as tensors
        image = TF.rotate(image, self.angle, interpolation=TF.InterpolationMode.NEAREST, expand=False)
        mask = TF.rotate(mask, self.angle, interpolation=TF.InterpolationMode.NEAREST, expand=False)

        return {'image': image, 'mask': mask}
    
 
class Rescale(object):
    """Rescale the image in a sample to a given size.
    
    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            self.output_size = output_size

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        
        # Resize image and mask
        image = TF.resize(image, self.output_size, interpolation=TF.InterpolationMode.BILINEAR)
        mask = TF.resize(mask, self.output_size, interpolation=TF.InterpolationMode.NEAREST)
        
        return {'image': image, 'mask': mask}
    
# class Rescale(object):
#     """Rescale the image in a sample to a given size.

#     Args:
#         output_size (tuple or int): Desired output size. If tuple, output is
#             matched to output_size. If int, smaller of image edges is matched
#             to output_size keeping aspect ratio the same.
#     """

#     def __init__(self):
#         # assert isinstance(output_size, (int, tuple))
#         self.new_h, self.new_w = (128, 128)

#     def __call__(self, sample):
#         image, mask = sample['image'], sample['mask']
        
#         # Rotate image and mask as tensors
#         image = TF.resize(image, [self.new_h, self.new_w])
#         mask = TF.resize(mask, [self.new_h, self.new_w])
        
#         return {'image': image, 'mask': mask}           
    
class BlurData(Dataset):
    # Constructor
    def __init__(self, root_dir, list_image, list_mask, transform = None):
        self.list_image = list_image
        self.list_mask = list_mask
        self.root_dir = root_dir
        self.transform = transform
        self.len = len(list_image)
    # Getting the data
    def __getitem__(self, index):
        # print(self.list_image[index])
        img_path = os.path.join(self.root_dir, "image", self.list_image[index])
        image = Image.open(img_path).convert("RGB")
        mask_path = os.path.join(self.root_dir, "mask", self.list_mask[index])
        mask = Image.open(mask_path).convert("RGB")
        sample = {'image': image, 'mask': mask}

        if self.transform:
            sample = self.transform(sample)

        return sample
           
    # Getting length of the data
    def __len__(self):
        return self.len


if __name__ == "__main__":
    root_dir = "data/train"
    list_image = os.listdir(os.path.join(root_dir, "image"))
    list_mask = os.listdir(os.path.join(root_dir, "mask"))
    
    bd = BlurData(root_dir, list_image, list_mask)
    
    sample = bd.__getitem__(3) 
    
    img_o = sample['image']
    mask_o = sample['mask']
    
    transform = transforms.Compose([Rotate(-30)])
    rotated_sample = transform(sample)
    img = rotated_sample['image']
    mask = rotated_sample['mask']
    # img = TF.to_pil_image(img.to("cpu"))
    # mask = TF.to_pil_image(mask.to("cpu"))
    
    img_o.show()
    mask_o.show()
    
    img.show()
    mask.show()