#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 13:46:53 2024

@author: becky
"""

import argparse
import os
import random
import os
# # Set CUBLAS_WORKSPACE_CONFIG for deterministic CuBLAS operations
# os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from model import *
# from IPython.display import HTML

from dataset import *



# Number of channels in the training images. For color images this is 3
nc = 3
# Size of z latent vector (i.e. size of generator input)
nz = 3
# Beta1 hyperparameter for Adam optimizers
beta1 = 0.5

def datalist(root_dir):
    list_image = os.listdir(os.path.join(root_dir, "image"))
    list_mask = os.listdir(os.path.join(root_dir, "mask"))
    
    temp = list(zip(list_image, list_mask))
    random.shuffle(temp)
    res1, res2 = zip(*temp)
    # res1 and res2 come out as tuples, and so must be converted to lists.
    images, masks = list(res1), list(res2)
    return images, masks

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
        
def validate_model(model, dataloader, criterion, lpips_vgg):
    model.eval()  # Set the model to evaluation mode
    total_lpips = 0.0
    with torch.no_grad():  # We don't need to track gradients during validation
        for i, data in enumerate(dataloader):
            inputs, targets = data['image'].to(device), data['mask'].to(device)
            main_outputs = model(inputs)
            outputs = main_outputs['out']
            loss = criterion(outputs, targets)

            # Calculate LPIPS for the current batch
            batch_lpips = lpips_vgg(outputs, targets).mean()
            total_lpips += batch_lpips.item()

    average_lpips = total_lpips / len(dataloader)
    print(f"Average LPIPS score for this epoch: {average_lpips}")
    return average_lpips


# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Parameters
num_epochs = 30
batch_size = 8
learning_rate = 0.0001

# Transformations
transform = transforms.Compose([
    ToTensor(),
    # Rotate(-30),
    #Rescale((224, 224))
])

# Load dataset
total_dir = 'data/train'  # Update this path
val_dir = "data/val"

total_images, total_masks = datalist(total_dir)
temp = list(zip(total_images, total_masks))
random.shuffle(temp)
res1, res2 = zip(*temp)
# res1 and res2 come out as tuples, and so must be converted to lists.
total_images_shuffle, total_masks_shuffle = list(res1), list(res2)

total_num = len(total_images)
train_num = int(total_num*0.8)
test_num = (total_num*0.1)
val_num = int(total_num*0.1)

train_images = total_images_shuffle[0:train_num]
train_masks = total_masks_shuffle[0:train_num]

val_images = total_images_shuffle[train_num+1:train_num+val_num+1]
val_masks = total_masks_shuffle[train_num+1:train_num+val_num+1]
# create the train and test datasets
train_dataset = BlurData(total_dir, train_images, train_masks, transform = transform)
val_dataset = BlurData(total_dir, val_images, val_masks, transform = transform)

print(f"[INFO] found {len(train_dataset)} examples in the training set...")
print(f"[INFO] found {len(val_dataset)} examples in the test set...")

# DataLoader
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

model_path = os.path.join("checkpoints", 'checkpoins_netG_{}.pth'.format(29))
netG = torch.load(model_path).to(device)



# Grab a batch of real images from the dataloader
real_batch = next(iter(train_dataloader))

fixed_noise = real_batch["image"]
# fake = netG(fixed_noise)["out"].detach().cpu()
netG = netG.to(device)
fixed_noise = fixed_noise.to(device)
fake = netG(fixed_noise)["out"].detach().cpu()

img_list = real_batch["mask"] 

real_images = np.transpose(vutils.make_grid(real_batch["mask"].to(device)[:64][1], padding=5, normalize=True).cpu(),(1,2,0))
noise = np.transpose(vutils.make_grid(fake.to(device)[:64][1], padding=5, normalize=True).cpu(),(1,2,0))
noise_image = np.transpose(vutils.make_grid(real_batch["image"].to(device)[:64][1], padding=5, normalize=True).cpu(),(1,2,0))
denoise_image = np.transpose(vutils.make_grid((real_batch["image"].to(device)[:64][1]-fake.to(device)[:64][1]), padding=5, normalize=True).cpu(),(1,2,0)) 

# Plot the real images
plt.figure(figsize=(4,4))
plt.subplot(1,2,1)
plt.axis("off")
plt.title("Real Images")
plt.imshow(real_images)

# Plot the fake images from the last epoch
plt.subplot(1,2,2)
plt.axis("off")
plt.title("Fake Images")
plt.imshow(noise)
# plt.imshow(np.transpose(fake[-1],(1,2,0)))
plt.show()

# Plot the real images
plt.figure(figsize=(4,4))
plt.subplot(1,2,1)
plt.axis("off")
plt.title("Real Images")
plt.imshow(real_images)

# Plot the fake images from the last epoch
plt.subplot(1,2,2)
plt.axis("off")
plt.title("Denoised Images")
plt.imshow(denoise_image)
# plt.imshow(np.transpose(fake[-1],(1,2,0)))
plt.show()

