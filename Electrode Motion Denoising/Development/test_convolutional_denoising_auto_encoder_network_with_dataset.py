import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import math
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import h5py
from ecgdetectors import Detectors

# Convolutional Auto-Encoder.
class ConvolutionalAutoEncoder(nn.Module):
    
    # Initialization Function.
    def __init__(self):
        super().__init__()
        
        # First layer is a 1D conv layer.
        self.conv1 = nn.Conv1d(1, 16, 3, padding=1)  # 1 input channel, 16 output channels (Filters), kernel size 3 (each filter looks at 3 consecutive elements, padding = 1)
        self.conv2 = nn.Conv1d(16, 8, 3, padding=1) # 16 input channels, 8 filters, kernel size 3. padding = 1
        self.conv3 = nn.Conv1d(8, 1, 3, padding=1) # 8 input channels, 1 filter, kernel size 3, padding = 1.
        self.conv4 = nn.Conv1d(1, 1, 1500, padding=0)  # Adjust padding to avoid input size issues. Kernel now looks at final segment. (1500 samples)
        self.acti = nn.ReLU() # Set the activation function as Rectified Linear Unit.
        self.out = nn.Sigmoid() # Set the output layer as a sigmoid function.

    # Set the forward pass.
    def forward(self, x):
        
        # Pass the signal through 1st layer.
        x = self.conv1(x)
        
        # Pass signal through activation function (ReLU)
        x = self.acti(x)
        
        # Pass the signal through the second layer.
        x = self.conv2(x)
        
        # Pass through activation.
        x = self.acti(x)
        
        # Pass through 3rd layer.
        x = self.conv3(x)
        
        # Pass through activation.
        x = self.acti(x)
        
        # Pass through 4th layer.
        x = self.conv4(x)
        
        # Flatten all dimensions except the batch
        x = torch.flatten(x, 1)
        
        # Pass through output layer (Sigmoid)
        out = self.out(x)
        
        return out