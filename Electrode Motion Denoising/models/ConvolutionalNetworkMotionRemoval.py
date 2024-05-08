import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.io import loadmat

# Constants
BATCH_SIZE = 20
EPOCHS = 10
LEARNING_RATE = 0.0002
CLEAN_DATA_PATH = '/Users/benrussell/Frameworks/ElectrodeMotionDenoising/Electrode Motion Denoising/models/datastore/cleanSignals'
NOISY_DATA_PATH = '/Users/benrussell/Frameworks/ElectrodeMotionDenoising/Electrode Motion Denoising/models/datastore/noiseSignal'

# Check if CUDA is available, else use CPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Using: {torch.cuda.get_device_name(device) if torch.cuda.is_available() else "CPU"}')

# Function to load data from .mat files
def load_mat_data(file_path):
    # Load .mat file
    data = loadmat(file_path)
    
    # Assuming the data variable in .mat file is named 'data', change as necessary
    return data['data']

# Load in the data from .mat files
clean_data = load_mat_data(CLEAN_DATA_PATH)
noisy_data = load_mat_data(NOISY_DATA_PATH)

# Preprocess and split the data
def preprocess_data(clean_data, noisy_data, peaks):
    max_length = max(len(row) for row in peaks)
    x_result = np.array([np.pad(row, (0, max_length - len(row))) for row in peaks])

    combined_data = []
    for i in range(len(clean_data)):
        combined_data.append([clean_data[i], x_result[i]])

    arr_train, arr_test = train_test_split(combined_data, test_size=0.2, random_state=42, shuffle=False)
    nst_train, nst_test = train_test_split(noisy_data, test_size=0.2, random_state=42, shuffle=False)

    return arr_train, arr_test, nst_train, nst_test

arr_train, arr_test, nst_train, nst_test = preprocess_data(clean_data, noisy_data, peaks)

# Convert lists to DataLoader
arr_trainloader = DataLoader(arr_train, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
arr_testloader = DataLoader(arr_test, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
nst_trainloader = DataLoader(nst_train, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
nst_testloader = DataLoader(nst_test, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

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
        x = torch.flatten(x, 1)
        return self.out(x)

net = CNN().float().to(device)
optimizer = optim.RMSprop(net.parameters(), lr=LEARNING_RATE)
criterion = nn.MSELoss().to(device)

def train_model(model, criterion, optimizer, train_loader, epochs):
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs.float().to(device))
            loss = criterion(outputs, labels.float().to(device))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 1000 == 999:
                print(f'[{epoch + 1}, {i + 1}] loss: {running_loss / 1000:.3f}')
                running_loss = 0.0

train_model(net, criterion, optimizer, arr_trainloader, EPOCHS)

# Saving model
torch.save(net.state_dict(), './model_checkpoint.pt')

print(f'Finished Training with {len(arr_trainloader)} batches in train loader and {len(arr_testloader)} in test loader.')
