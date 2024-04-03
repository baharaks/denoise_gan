#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 20:18:23 2024

@author: becky
"""

import torch
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F

# Size of feature maps in discriminator
ndf = 32
nc = 3

def Generator(output_channels=3):
    # Load a pretrained DeepLabV3 model
    model = models.segmentation.deeplabv3_resnet101(pretrained=True, progress=False)
    
    # Adjust the output layer of the main classifier to match the desired output channels
    model.classifier[4] = nn.Conv2d(256, output_channels, kernel_size=(1, 1))
    
    # Remove the auxiliary classifier
    model.aux_classifier = None  # This disables the auxiliary classifier
    
    return model

class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # Assuming `nc` is the number of channels in the input images,
            # and `ndf` is the size of the feature maps in the discriminator.
            # Input is `(nc) x 256 x 256`
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # State size. `(ndf) x 128 x 128`
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # State size. `(ndf*2) x 64 x 64`
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # State size. `(ndf*4) x 32 x 32`
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # State size. `(ndf*8) x 16 x 16`
            nn.Conv2d(ndf * 8, ndf * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # State size. `(ndf*16) x 8 x 8`
            nn.Conv2d(ndf * 16, ndf * 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 32),
            nn.LeakyReLU(0.2, inplace=True),
            # State size. `(ndf*32) x 4 x 4`
            nn.Conv2d(ndf * 32, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
            # Output size. `1 x 1 x 1`, which is a single scalar value per image indicating real or fake
        )

    def forward(self, input):
        return self.main(input)
    
