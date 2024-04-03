#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 20:12:16 2024

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

# Number of workers for dataloader
workers = 2
# Size of z latent vector (i.e. size of generator input)
nz = 3
# Beta1 hyperparameter for Adam optimizers
beta1 = 0.5
# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1


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

total_images, total_masks = datalist(total_dir)
temp = list(zip(total_images, total_masks))
random.shuffle(temp)
res1, res2 = zip(*temp)

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

# uncomment this part to check the images
# real_batch = next(iter(train_dataloader))
# plt.figure(figsize=(8,8))
# plt.axis("off")
# plt.title("Training Images")
# plt.imshow(np.transpose(vutils.make_grid(real_batch["image"].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
# plt.show()

# Create the generator
netG = Generator(output_channels=3).to(device)

# Handle multi-GPU if desired
if (device.type == 'cuda') and (ngpu > 1):
    netG = nn.DataParallel(netG, list(range(ngpu)))

# Freeze all layers in the model
for param in netG.parameters():
    param.requires_grad = False

# Unfreeze the last layer - make it trainable
for param in netG.classifier[4].parameters():
    param.requires_grad = True

# # Print the model
# print(netG)

# Create the Discriminator
netD = Discriminator(ngpu).to(device)

# Handle multi-GPU if desired
if (device.type == 'cuda') and (ngpu > 1):
    netD = nn.DataParallel(netD, list(range(ngpu)))

# Apply the ``weights_init`` function to randomly initialize all weights
# like this: ``to mean=0, stdev=0.2``.
netD.apply(weights_init)

# # Print the model
# print(netD)

# Initialize the ``BCELoss`` function
criterion = nn.BCELoss()

# Create batch of latent vectors that we will use to visualize
#  the progression of the generator
fixed_noise = torch.randn(64, nz, 1, 1, device=device)

# Establish convention for real and fake labels during training
real_label = 1.
fake_label = 0.

# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

# Training Loop

# Lists to keep track of progress
img_list = []
G_losses = []
D_losses = []
iters = 0

print("Starting Training Loop...")
# For each epoch
for epoch in range(num_epochs):
    # For each batch in the dataloader
    for i, data in enumerate(train_dataloader, 0):
        noise, real_cpu = data['image'].to(device), data['mask'].to(device)
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## Train with all-real batch
        netD.zero_grad()
        # Format batch
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
        # Forward pass real batch through D
        output = netD(real_cpu).view(-1)
        # Calculate loss on all-real batch
        errD_real = criterion(output, label)
        # Calculate gradients for D in backward pass
        errD_real.backward()
        D_x = output.mean().item()

        ## Train with all-noise batch
        fake = netG(noise)
        label.fill_(fake_label)
        # Classify all noise batch with D
        output = netD(fake["out"].detach()).view(-1)
        # Calculate D's loss on the all-noise batch
        errD_fake = criterion(output, label)
        # Calculate the gradients for this batch, accumulated (summed) with previous gradients
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        # Compute error of D as sum over the noise and the sharp batches
        errD = errD_real + errD_fake
        # Update D
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = netD(fake["out"]).view(-1)
        # Calculate G's loss based on this output
        errG = criterion(output, label)
        # Calculate gradients for G
        errG.backward()
        D_G_z2 = output.mean().item()
        # Update G
        optimizerG.step()

        # Output training stats
        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, num_epochs, i, len(train_dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())

        iters += 1
        
    model_path = os.path.join("checkpoints", 'checkpoins_netD_{}.pth'.format(epoch))
    torch.save(netD, model_path)
    
    model_path = os.path.join("checkpoints", 'checkpoins_netG_{}.pth'.format(epoch))
    torch.save(netG, model_path)

plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses,label="G")
plt.plot(D_losses,label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()

