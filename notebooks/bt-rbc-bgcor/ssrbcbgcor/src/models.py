# -*- coding: utf-8 -*-
"""
Created on Fri 29 15:46:32 2022

example: ResNet18 architecture

@author: tadahaya
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class Block(nn.Module):
    """
    A residual block consisting of three convolutional layers with skip connections.

    Args:
        channel_in (int): Number of input channels.
        channel_out (int): Number of output channels.
        stride (int, optional): Stride for the first convolutional layer. Defaults to 1.
    """
    def __init__(self, channel_in, channel_out, stride=1):
        super().__init__()

        # 1st convolution (3x3)
        self.conv1 = nn.Conv2d(
            channel_in, channel_out, kernel_size=3, stride=stride, padding=1, bias=False
            )
        # bias=False: Batch normalization layer is used after the convolutional layer
        self.bn1 = nn.BatchNorm2d(channel_out) # Batch normalization

        # 2nd convolution (3 x 3)
        self.conv2 = nn.Conv2d(
            channel_out, channel_out, kernel_size=(3,3), stride=1, padding=1, bias=False
            )
        self.bn2 = nn.BatchNorm2d(channel_out) # Batch normalization

        # Shortcut connection (identity mapping or projection)
        self.shortcut = nn.Sequential()
        if stride != 1 or channel_in != channel_out:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    channel_in, channel_out, kernel_size=(1,1), stride=stride, padding=0, bias=False
                    ),
                nn.BatchNorm2d(channel_out)
            )


    def forward(self,x):
        """
        Forward pass through the residual block.
        
        Args:
            x (Tensor): Input tensor of shape (batch_size, channel_in, height, width).

        Returns:
            Tensor: Output tensor of shape (batch_size, channel_out, height, width).
        """
        shortcut = self.shortcut(x) # Skip connection
        h = self.conv1(x) # Shape: (batch_size, C, H, W)
        h = self.bn1(h) # Batch normalization
        h = F.relu(h) # ReLU activation
        h = self.conv2(h) # Shape: (batch_size, C, H, W)
        h = self.bn2(h) # Batch normalization
        h += shortcut # Skip connection
        h = F.relu(h) # ReLU activation AFTER skip connection
        return h


class MyNet(nn.Module):
    """
    A ResNet-like neural network for classification tasks.

    Args:
        output_dim (int): Number of output dimensions for the final classification layer.
    """
    def __init__(self, output_dim):
        super().__init__()

        # Initial convolutional layer
        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=(7,7), stride=(2,2), padding=3, bias=False
            ) # 7 x 7 convolution, (B, 3, H, W) -> (B, 64, H/2, W/2)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True) # In-place ReLU activation for memory efficiency
        self.maxpool = nn.MaxPool2d(kernel_size=(3,3), stride=(2,2), padding=1)
        # (B, 64, H/2, W/2) -> (B, 64, H/4, W/4)
        
        # Residual blocks
        self.block0 = self._building_block(64, 64, stride=1, num_blocks=2)
        # (B, 64, H/4, W/4) -> (B, 64, H/4, W/4), keep the same size
        self.block1 = self._building_block(64, 128, stride=2, num_blocks=2)
        # (B, 64, H/4, W/4) -> (B, 128, H/8, W/8), downsample
        self.block2 = self._building_block(128, 256, stride=2, num_blocks=2)
        # (B, 128, H/8, W/8) -> (B, 256, H/16, W/16), downsample
        self.block3 = self._building_block(256, 512, stride=2, num_blocks=2)
        # (B, 256, H/16, W/16) -> (B, 512, H/32, W/32), downsample

        # Final layers
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) # Global average pooling
        # (B, 512, H/32, W/32) -> (B, 512, 1, 1)
        self.fc = nn.Linear(512, output_dim)


    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (Tensor): Input tensor of shape (batch_size, 3, height, width).

        Returns:
            Tensor: Output tensor of shape (batch_size, output_dim).
        """
        # Initial convolutional layer
        h = self.conv1(x) # (B, 3, H, W) -> (B, 64, H/2, W/2)
        h = self.bn1(h)
        h = self.relu(h)
        h = self.maxpool(h) # (B, 64, H/2, W/2) -> (B, 64, H/4, W/4)
        # Residual blocks
        h = self.block0(h) # (B, 64, H/4, W/4) -> (B, 64, H/4, W/4)
        h = self.block1(h) # (B, 64, H/4, W/4) -> (B, 128, H/8, W/8)
        h = self.block2(h) # (B, 128, H/8, W/8) -> (B, 256, H/16, W/16)
        h = self.block3(h) # (B, 256, H/16, W/16) -> (B, 512, H/32, W/32)
        # Final layers
        h = self.avgpool(h) # (B, 512, H/32, W/32) -> (B, 512, 1, 1)
        h = h.view(h.size(0), -1) # (B, 512, 1, 1) -> (B, 512)
        h = self.fc(h) # Shape: (batch_size, output_dim)
        return h


    def _building_block(self, channel_in, channel_out, stride, num_blocks):
        """
        Helper function to create a residual block.

        Args:
            channel_out (int): Number of input channels.
            channel_in (int): Number of output channels.
            stride (int): Stride for the first convolutional layer.
            num_blocks (int): Number of residual blocks to create.

        Returns:
            Block: A residual block instance.
        """
        layers = []
        layers.append(Block(channel_in, channel_out, stride)) # First block with stride
        for _ in range(1, num_blocks):
            layers.append(Block(channel_out, channel_out, stride=1)) # Remaining blocks with stride=1
        return nn.Sequential(*layers)