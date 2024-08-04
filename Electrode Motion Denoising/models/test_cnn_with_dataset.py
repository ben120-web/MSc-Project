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

# Check if GPU is available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
userSelectTrain = False

if torch.cuda.is_available():
    cuda = True
    print('Using: ' + str(torch.cuda.get_device_name(device)))
elif torch.backends.mps.is_available():
    device = torch.device('mps')
    cuda = True
    print('Using: MPS')
else:
    cuda = False
    print('Using: CPU')

Tensor = torch.cuda.FloatTensor if cuda and torch.cuda.is_available() else torch.FloatTensor if cuda and torch.backends.mps.is_available() else torch.FloatTensor

# Set the Peak detectors to work at 500Hz
detectors = Detectors(500)

# Assuming you have mounted your Google Drive and set the correct path
clean_signals_path = '/Users/benrussell/Library/CloudStorage/GoogleDrive-rben3625@gmail.com/My Drive/cleanSignals/'
noisy_signals_path = '/Users/benrussell/Library/CloudStorage/GoogleDrive-rben3625@gmail.com/My Drive/noisySignals/'

# Function to load HDF5 files
def load_h5_file(file_path):
    try:
        with h5py.File(file_path, 'r') as f:
            return f['ecgSignal'][:]
    except OSError as e:
        print("Unable to open file" + file_path)
        return None

# Function to get Root Mean Square
def get_rms(records):
    return math.sqrt(sum([x ** 2 for x in records]) / len(records))

# Custom Loss Function
def lossfcn(y_true, y_pred, peaks, a=20):
    criterion = nn.MSELoss().to(device)
    alpha = a
    loss = 0.0
    R = 0.0
    for x, y, z in zip(y_pred, y_true, peaks):
        qrs_loss = []

        # Remove Padding from NN
        z = z[z.nonzero()]
        for qrs in z:
            max_ind = qrs + 1

            if max_ind < 35:
                qrs_loss.append(criterion(x[:max_ind + 37], y[:max_ind + 37]))
            elif max_ind > 1243:
                qrs_loss.append(criterion(x[max_ind - 36:], y[max_ind - 36:]))
            else:
                qrs_loss.append(criterion(x[max_ind - 36:max_ind + 37], y[max_ind - 36:max_ind + 37]))

        R_loss = alpha * (torch.mean(torch.tensor(qrs_loss)))
        if math.isnan(R_loss):
            R_loss = 0
        R += R_loss

    loss = criterion(y_true, y_pred) + torch.mean(torch.tensor(R))
    return loss

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# ECG Dataset
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

# Convolutional Neural Network Model definition
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

if __name__ == '__main__':
    print("Starting script execution...")

    # Load clean signals
    clean_signals = {}
    for file in glob.glob(os.path.join(clean_signals_path, '*.h5')):
        signal_number = os.path.basename(file).split('_')[1]
        clean_signals[signal_number] = load_h5_file(file)

    # Load noisy signals and group by signal number
    noisy_signals = {snr: {} for snr in ['SNR18', 'SNR24']} #'SNR0', 'SNR6', 'SNR12', 

    # Loop through each SNR level.
    for snr in noisy_signals.keys():
        # Set the path to the specific SNR under test.
        snr_path = os.path.join(noisy_signals_path, snr)

        # Loop through each file in the directory of the SNR.
        for file in glob.glob(os.path.join(snr_path, '*.h5')):
            # Extract the signal number from the file name.
            signal_number = os.path.basename(file).split('-')[0].split('_')[1]

            # If the signal number is not in noisy signals, set as empty.
            if signal_number not in noisy_signals[snr]:
                noisy_signals[snr][signal_number] = []

            # Append the noisy signal to the dictionary.
            noisy_signals[snr][signal_number].append(load_h5_file(file)) 

    # Verify the structure
    print("Clean signals loaded:", len(clean_signals))

    for snr in noisy_signals:
        print(f"Noisy signals loaded for {snr}: {len(noisy_signals[snr])}") 

    # Create the dataset and dataloader
    dataset = ECGDataset(clean_signals, noisy_signals)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)

    net = CNN().float().to(device)
    criterion = nn.MSELoss().to(device)
    optimizer = optim.RMSprop(net.parameters(), lr=0.0002)

    # Training (Only train if specified.)
    if userSelectTrain:
        for epoch in range(10):  # loop over the dataset multiple times
            print("=========== EPOCH " + str(epoch) + " ===========")
            running_loss = 0.0

            # Iterate over all clean signals
            for signal_number in clean_signals:
                clean_signal = clean_signals[signal_number][0].astype(np.float32)  # Ensure the clean signal is in float32
                clean_signal = torch.tensor(clean_signal).to(device).unsqueeze(0)  # Add batch dimension

                # Iterate over all noisy variants for the current clean signal
                for snr in noisy_signals:
                    if signal_number in noisy_signals[snr]:  # Check if the signal number exists
                        for noisy_variant in noisy_signals[snr][signal_number]:
                            noisy_signal = torch.tensor(noisy_variant.astype(np.float32)).to(device).unsqueeze(0)  # Ensure the noisy signal is in float32 and add batch dimension

                            optimizer.zero_grad()
                            outputs = net(noisy_signal)
                            loss = criterion(outputs, clean_signal)
                            loss.backward()
                            optimizer.step()
                            running_loss += loss.item()

            print(f'Epoch [{epoch + 1}/10], Loss: {running_loss / len(clean_signals)}')

        print('Finished Training')

        # Save the model
        torch.save(net.state_dict(), './model_weights.pt')
        
    else:
        print('Using pre-trained model')
        
        # Load the saved model
        net.load_state_dict(torch.load('./model_weights.pt'))
        net.eval()

    # Evaluation - Plot only 5 signals
    with torch.no_grad():
        plot_count = 0
        for clean_signal, noisy_signals in dataloader:
            print(f'clean_signal type: {type(clean_signal)}, shape: {clean_signal[0].shape}')
            print(f'noisy_signals type: {type(noisy_signals)}, length: {len(noisy_signals)}')

            clean_signal = clean_signal[0].float().unsqueeze(0).to(device)  # Shape: (1, 1, 15000)
            
            for noisy_signal in noisy_signals:
                noisy_signal = noisy_signal[0].float().unsqueeze(0).to(device)  # Shape: (1, 1, 15000)
                
                outputs = net(noisy_signal)
                
                plt.figure()
                plt.plot(noisy_signal.cpu().numpy().flatten(), label='noisy')
                plt.plot(clean_signal.cpu().numpy().flatten(), label='clean')
                plt.plot(outputs.cpu().numpy().flatten(), label='denoised')
                plt.legend()
                plt.show()
                
                plot_count += 1
                if plot_count >= 5:
                    break
            if plot_count >= 5:
                break

    # Save the model
    torch.save(net.state_dict(), './model_weights.pt')
    print("Script execution completed.")
