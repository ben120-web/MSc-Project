import torch
import torch.nn as nn
import torch.optim as optim
import math
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split # type: ignore
from signal_dataset_loader import get_dataloader  # Import the data loader function

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

if torch.cuda.is_available():
    cuda = True
    print('Using: ' + str(torch.cuda.get_device_name(device)))
else:
    cuda = False
    print('Using: CPU')

# If GPU is available, use corresponding FloatTensor. If not, use the CPU tensor.
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# Example to create DataLoaders for training and testing
clean_signals_dir = r'C:\B-Secur\MSc Project\ElectrodeMotionionDenoisingFramework\Electrode Motion Denoising\models\datastore\trainingDataSet\cleanSignals'
noisy_signals_dir = r'C:\B-Secur\MSc Project\ElectrodeMotionionDenoisingFramework\Electrode Motion Denoising\models\datastore\trainingDataSet\noisySignals'
snr_levels = ['SNR0', 'SNR6', 'SNR12', 'SNR18', 'SNR24']  # List all desired SNR levels

train_dataloader = get_dataloader(clean_signals_dir, noisy_signals_dir, snr_levels, batch_size=20, shuffle=True, num_workers=2)

def get_rms(records):
    return math.sqrt(sum([x ** 2 for x in records]) / len(records))

peaks = torch.load('./train_peaks.pt')

clean_data = torch.load('./clean_train_data_2dB.pt')
noisy_data = torch.load('./noisy_train_data_2dB.pt')

clean_train_data = clean_data[:109000]
noisy_train_data = noisy_data[:109000]
peaks_train = peaks[:109000]

clean_test_data = clean_data[109000:136000]
noisy_test_data = noisy_data[109000:136000]
peaks_test = peaks[109000:136000]

batch_size = 20

######## Training_Set ###########
#
# Pad the arrays to max length of peaks
#
#################################
max_length = max(len(row) for row in peaks_train)

x_result = np.array([np.pad(row, (0, max_length - len(row))) for row in peaks_train])

# Create Training dataset with peaks
_data = []

for i in range(len(clean_train_data)):
    _data.append([clean_train_data[i], x_result[i]])

# preparing trainLoaders 
arr_trainloader = DataLoader(_data, batch_size=batch_size, shuffle=False, num_workers=2)
nst_trainloader = DataLoader(noisy_train_data, batch_size=batch_size, shuffle=False, num_workers=2)

######## Test_Set ###############
#
# Pad the arrays to max length of peaks
#
#################################
max_length = max(len(row) for row in peaks_test)

x_result = np.array([np.pad(row, (0, max_length - len(row))) for row in peaks_test])

# Create Testing dataset with peaks
_data = []

for i in range(len(clean_test_data)):
    _data.append([clean_test_data[i], x_result[i]])

# preparing testLoaders 
arr_testloader = DataLoader(_data, batch_size=batch_size, shuffle=False, num_workers=2)
nst_testloader = DataLoader(noisy_test_data, batch_size=batch_size, shuffle=False, num_workers=2)
print(len(arr_trainloader), len(arr_testloader))

clean_data = clean_data[:136000]
noisy_data = noisy_data[:136000]
peaks = peaks[:136000]

batch_size = 20
# Pad the arrays to max length of peaks
max_length = max(len(row) for row in peaks)

x_result = np.array([np.pad(row, (0, max_length - len(row))) for row in peaks])

# Create Training dataset with peaks
_data = []

for i in range(len(clean_data)):
    _data.append([clean_data[i], x_result[i]])

# preparing training dataset and testing dataset
arr_train, arr_test = train_test_split(_data, test_size=0.2, random_state=42, shuffle=False)
nst_train, nst_test = train_test_split(noisy_data, test_size=0.2, random_state=42, shuffle=False)

del _data, noisy_data, clean_data

# preparing trainLoaders 
arr_trainloader = DataLoader(arr_train, batch_size=batch_size, shuffle=False, num_workers=2)
nst_trainloader = DataLoader(nst_train, batch_size=batch_size, shuffle=False, num_workers=2)

# preparing testLoaders 
arr_testloader = DataLoader(arr_test, batch_size=batch_size, shuffle=False, num_workers=2)
nst_testloader = DataLoader(nst_test, batch_size=batch_size, shuffle=False, num_workers=2)

print(len(arr_trainloader), len(arr_testloader))

## RCNN MODEL.
class RCNN(nn.Module):
    def __init__(self, input_size):
        super(RCNN, self).__init__()

        self.input_size = input_size

        self.cv1_k = 3
        self.cv1_s = 1
        self.cv1_out = int(((self.input_size - self.cv1_k) / self.cv1_s) + 1)

        self.cv2_k = 3
        self.cv2_s = 1
        self.cv2_out = int(((self.cv1_out - self.cv2_k) / self.cv2_s) + 1)

        self.cv3_k = 5
        self.cv3_s = 1
        self.cv3_out = int(((self.cv2_out - self.cv3_k) / self.cv3_s) + 1)

        self.cv4_k = 5
        self.cv4_s = 1
        self.cv4_out = int(((self.cv3_out - self.cv4_k) / self.cv4_s) + 1)

        self.layer_1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=3, kernel_size=(3)),
            nn.BatchNorm1d(num_features=3),
            nn.ReLU(inplace=True),
            nn.AvgPool1d(kernel_size=1)
        )

        self.layer_2 = nn.Sequential(
            nn.Conv1d(in_channels=3, out_channels=5, kernel_size=(3)),
            nn.BatchNorm1d(num_features=5),
            nn.ReLU(inplace=True),
            nn.AvgPool1d(kernel_size=1)
        )

        self.layer_3 = nn.Sequential(
            nn.Conv1d(in_channels=5, out_channels=3, kernel_size=(5)),
            nn.BatchNorm1d(num_features=3),
            nn.ReLU(inplace=True)
        )

        self.layer_4 = nn.Sequential(
            nn.Conv1d(in_channels=3, out_channels=1, kernel_size=(5)),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False)
        )

        self.layer_5 = nn.Sequential(
            nn.Linear(self.cv4_out, 1080),  # FC Layer
            nn.Linear(1080, 1080)  # Regression
        )

    def forward(self, x):
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)
        x = x.view(x.size(0), -1)
        x = self.layer_5(x)
        return x

## CNN MODEL 
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
        x = (self.conv1(x))
        x = (self.acti(x))
        x = (self.conv2(x))
        x = (self.acti(x))
        x = (self.conv3(x))
        x = (self.acti(x))
        x = (self.conv4(x))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        out = self.out(x)
        return out

def lossfcn(y_true, y_pred, peaks, a=20):
    criterion = nn.MSELoss().to(device)
    alpha = a
    loss = 0.0
    R = 0.0

    for x, y, z in zip(y_pred, y_true, peaks):
        qrs_loss = []

