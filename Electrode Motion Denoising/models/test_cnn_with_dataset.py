import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.variable import Variable
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import math

from sklearn.model_selection import train_test_split

import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import math 
import json as js
import h5py
import pandas as pd
from sklearn.model_selection import train_test_split

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

if torch.cuda.is_available():
  cuda = True
  print('Using: ' +str(torch.cuda.get_device_name(device)))
else:
  cuda = False
  print('Using: CPU')

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

def get_rms(records):
    return math.sqrt(sum([x ** 2 for x in records]) / len(records))

# Assuming you have mounted your Google Drive and set the correct path
clean_signals_path = '/Users/benrussell/Library/CloudStorage/GoogleDrive-rben3625@gmail.com/My Drive/cleanSignals/'
noisy_signals_path = '/Users/benrussell/Library/CloudStorage/GoogleDrive-rben3625@gmail.com/My Drive/noisySignals/'

# Function to load HDF5 files
def load_h5_file(file_path):
    with h5py.File(file_path, 'r') as f:
        return f['ecgSignal'][:]

# Load clean signals
clean_signals = {}
for file in glob.glob(os.path.join(clean_signals_path, '*.h5')):
    signal_number = os.path.basename(file).split('_')[1]
    clean_signals[signal_number] = load_h5_file(file)

# Load noisy signals and group by signal number
noisy_signals = {snr: {} for snr in ['SNR0', 'SNR6', 'SNR12', 'SNR18', 'SNR24']}

for snr in noisy_signals.keys():
    snr_path = os.path.join(noisy_signals_path, snr)
    for file in glob.glob(os.path.join(snr_path, '*.h5')):
        signal_number = os.path.basename(file).split('-')[0].split('_')[1]
        if signal_number not in noisy_signals[snr]:
            noisy_signals[snr][signal_number] = []
        noisy_signals[snr][signal_number].append(load_h5_file(file))

# Verify the structure
print("Clean signals loaded:", len(clean_signals))
for snr in noisy_signals:
    print(f"Noisy signals loaded for {snr}: {len(noisy_signals[snr])}")
    
class ECGDataset(Dataset):
    def __init__(self, clean_signals, noisy_signals):
        self.clean_signals = clean_signals
        self.noisy_signals = noisy_signals
        self.signal_numbers = list(clean_signals.keys())

    def __len__(self):
        return len(self.clean_signals)

    def __getitem__(self, idx):
        signal_number = self.signal_numbers[idx]
        clean_signal = self.clean_signals[signal_number]
        noisy_signals_for_number = []
        for snr in self.noisy_signals:
            noisy_signals_for_number.extend(self.noisy_signals[snr].get(signal_number, []))
        return clean_signal, noisy_signals_for_number

# Create the dataset and dataloader
dataset = ECGDataset(clean_signals, noisy_signals)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)

# Model definition
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 16, 3, padding='same')
        self.conv2 = nn.Conv1d(16, 8, 3, padding='same')
        self.conv3 = nn.Conv1d(8, 1, 3, padding='same')
        self.conv4 = nn.Conv1d(1, 1, 1080, padding='same')
        self.acti = nn.ReLU()
        self.out = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.acti(x)
        x = self.conv2(x)
        x = self.acti(x)
        x = self.conv3(x)
        x = self.acti(x)
        x = self.conv4(x)
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        out = self.out(x)
        return out

net = CNN().float().to(device)
criterion = nn.MSELoss().to(device)
optimizer = optim.RMSprop(net.parameters(), lr=0.0002)

# Training
for epoch in range(10):  # loop over the dataset multiple times
    print("=========== EPOCH " + str(epoch) + " ===========")
    running_loss = 0.0

    for clean_signal, noisy_signal in dataloader:
        clean_signal = clean_signal.to(device).float()
        noisy_signal = torch.tensor(noisy_signal).to(device).float().unsqueeze(0)  # Add batch dimension

        optimizer.zero_grad()
        outputs = net(noisy_signal[:, None, :])
        loss = criterion(outputs, clean_signal[:, None, :])
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f'Epoch [{epoch + 1}/10], Loss: {running_loss/len(dataloader)}')

print('Finished Training')

# Evaluation
net.eval()
with torch.no_grad():
    for clean_signal, noisy_signal in dataloader:
        clean_signal = clean_signal.to(device).float()
        noisy_signal = torch.tensor(noisy_signal).to(device).float().unsqueeze(0)  # Add batch dimension
        outputs = net(noisy_signal[:, None, :])
        plt.figure()
        plt.plot(noisy_signal.cpu().numpy().flatten(), label='noisy')
        plt.plot(clean_signal.cpu().numpy().flatten(), label='clean')
        plt.plot(outputs.cpu().numpy().flatten(), label='denoised')
        plt.legend()
        plt.show()

# Save the model
torch.save(net.state_dict(), './model_weights.pt')