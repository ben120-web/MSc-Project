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

# USER SELECTION (Set to true if you want to train a new model. False will use existing weights and bias)
userSelectTrain = True

# Let's define a device to use. (GPU if desktop being used, MPS if Mac Book being used, CPU otherwise.)
if torch.cuda.is_available():
    cuda = True
    print('Using: ' + str(torch.cuda.get_deviceName(device)))
elif torch.backends.mps.is_available():
    device = torch.device('mps')
    cuda = True
    print('Using: MPS')
else:
    cuda = False
    print('Using: CPU')

# Set the Peak detectors to work at 500Hz.
detectors = Detectors(500)

# Set path to google drive. (This needs mounted to your machine.)
clean_signals_path = '/Users/benrussell/Library/CloudStorage/GoogleDrive-rben3625@gmail.com/My Drive/cleanSignals/'
noisy_signals_path = '/Users/benrussell/Library/CloudStorage/GoogleDrive-rben3625@gmail.com/My Drive/noisySignals/'

## HELPER FUNCTIONS

# Function to load HDF5 files.
def load_h5_file(file_path):
    try:
        with h5py.File(file_path, 'r') as f:
            data = f['ecgSignal'][:]
            print(f"Loaded {file_path}, shape: {data.shape}")
            return data
    except OSError as e:
        print("Unable to open file" + file_path)
        return None

# Function to get Root Mean Square. (This is used as a loss function.)
def get_rms(records):
    return math.sqrt(sum([x ** 2 for x in records]) / len(records))

# Custom Loss Function (This calculates the Mean Square Error around Heartbeats.)
def lossfcn(y_true, y_pred, peaks, a = 20):
    
    # Set the Criterion to use Mean Square Error.
    criterion = nn.MSELoss().to(device)
    alpha = a
    loss = 0.0
    R = 0.0
    
    # Loop through the predicted denoised signal, the correct clean signal and the peak locations.
    for x, y, z in zip(y_pred, y_true, peaks):
        
        # Initialise the loss around the QRS.
        qrs_loss = []
        
        # Ensure z is a tensor and handle indices properly
        if z.dim() > 0:
            
            # FIGURE OUT WHAT THIS DOES.
            z = z[z.nonzero(as_tuple = True)].squeeze()

            # Loop through the QRS locations (Peaks)
            for qrs in z:
                
                # Get the last peak in the signal.
                max_ind =   qrs.item() + 1

                # If the last peak is less than 35, append
                if max_ind < 35:
                    qrs_loss.append(criterion(x[:max_ind + 37], y[:max_ind + 37]))
                elif max_ind > 1243:
                    qrs_loss.append(criterion(x[max_ind - 36:], y[max_ind - 36:]))
                else:
                    qrs_loss.append(criterion(x[max_ind - 36:max_ind + 37], y[max_ind - 36:max_ind + 37]))

            # Calculate the weighted loss around the R peak vicinity.
            R_loss = alpha * (torch.mean(torch.tensor(qrs_loss)))
            
            # If the R loss is NaN, set as 0.
            if math.isnan(R_loss):
                R_loss = 0
            
            # Update the total loss of the signal pair.
            R += R_loss
    
    # Update the overall loss as the MSE + the weighted MSE.
    loss = criterion(y_true, y_pred) + torch.mean(torch.tensor(R))
    
    # Return the custom loss for the signal pair.
    return loss

# Function to count parameters in a Deep Neural Network.
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# ECG Dataset
class ECGDataset(Dataset):
    
    # Initialisation Function
    def __init__(self, clean_signals, noisy_signals, segment_length = 1500):
        
        # Intitialise the clean signals, noisy signals, signal numbers and segment length.
        self.clean_signals = clean_signals
        self.noisy_signals = noisy_signals
        self.signal_numbers = list(clean_signals.keys())
        self.segment_length = segment_length

    # Function to get the total number of segments in a signal.
    def __len__(self):
        
        # Intialise total segments as 0.
        total_segments = 0
        
        # Loop through each signal.
        for signal_number in self.signal_numbers:
            
            # Get the length of the signal (15000)
            signal_length = self.clean_signals[signal_number].size
            
            # If the signal length is greater than segment length (Should always be the case.)
            if signal_length >= self.segment_length:
                
                # Calculate the number of 3 second segments (Should always be 10.)
                num_segments = signal_length // self.segment_length
                
                # Update the total number of segments variable.
                total_segments += num_segments * len(self.noisy_signals)
                
        # Print the total number of segments in the database.
        print(f"Total segments: {total_segments}")
        
        # Return the total number of segments as integer.
        return int(total_segments)

    # Function to get segment the clean and noisy signals.
    def __getitem__(self, idx):
        
        # Initialise a segment index variable.
        segment_index = idx
        
        # Loop through each signal.
        for signal_number in self.signal_numbers:
            
            # Get the length of the signal.
            signal_length = self.clean_signals[signal_number].size
            
            # If signal is greater than segment (Should be)
            if signal_length >= self.segment_length:
                
                # Get the number of segments.
                num_segments = signal_length // self.segment_length
                
                # Loop through each SNR value of the noisy signals.
                for snr in self.noisy_signals:
                    
                    # If the segment index is less than the number of segments (Should break at end of signal.)
                    if segment_index < num_segments:
                        
                        # Set the start index of the segment.
                        start_idx = segment_index * self.segment_length
                        
                        # Extract a 3 second segment from the clean signal.
                        clean_signal_segment = self.clean_signals[signal_number][start_idx : start_idx + self.segment_length]
                        
                        # Extract a 3 second segment from the noisy signal.
                        noisy_signal_segment = self.noisy_signals[snr][signal_number][0][start_idx:start_idx + self.segment_length]
                        
                        # Return the segments as tensors. (CHECK ALL COPIES ARE COVERED IN NOISY SIGNALS)
                        return torch.tensor(clean_signal_segment).unsqueeze(0), torch.tensor(noisy_signal_segment).unsqueeze(0)
                    
                    segment_index -= num_segments
                    
        # Raise an out of range error when end is met.
        raise IndexError("Index out of range")

# Convolutional Neural Network Model definition
class CNN(nn.Module):
    
    # Initialisation Funtion.
    def __init__(self):
        super().__init__()
        
        # First layer is a 1D conv layer.
        self.conv1 = nn.Conv1d(1, 16, 3, padding=1)  # 1 input channel, 16 output channels (Filters), kernel size 3 (each filter looks at 3 consecutive elements, padding = 1)
        self.conv2 = nn.Conv1d(16, 8, 3, padding=1) # 16 input channels, 8 filters, kernel size 3. padding = 1
        self.conv3 = nn.Conv1d(8, 1, 3, padding=1) # 8 input channels, 1 filter, kernel size 3, padding = 1.
        self.conv4 = nn.Conv1d(1, 1, 1500, padding=0)  # Adjust padding to avoid input size issues. Kernal now looks at final segment. (1500 samples)
        self.acti = nn.ReLU() # Set the activation function as Rectified Linear Unit.
        self.out = nn.Sigmoid() #  Set the outout layer as a sigmoid function.s

    # Set the forward pass.
    def forward(self, x):
        
        # Pass the signal through 1st layer.
        x = self.conv1(x)
        
        # Pass signal thorugh activation function (ReLU)
        x = self.acti(x)
        
        # Pass the signal through second layer.
        x = self.conv2(x)
        
        # Pass through activation.
        x = self.acti(x)
        
        # Pass through 3rd layer.
        x = self.conv3(x)
        
        # Pass through activation.
        x = self.acti(x)
        
        # Pass through 4th later.
        x = self.conv4(x)
        
        # Flatten all domesions except the batch
        x = torch.flatten(x, 1)
        
        # Pass through output layer (Sigmoid)
        out = self.out(x)
        
        return out

# Start processing (Data Loading, Training, Evalutation)
if __name__ == '__main__':
    print("Starting script execution...")

    # Load clean signals
    clean_signals = {}
    
    # Loop through each h5 file in directory.
    for file in glob.glob(os.path.join(clean_signals_path, '*.h5')):
        
        # Get the signal number. 
        signal_number = os.path.basename(file).split('_')[1]
        
        # Load in all h5 files based on signal number.
        clean_signals[signal_number] = load_h5_file(file)

    # Load noisy signals and group by signal number
    noisy_signals = {snr: {} for snr in ['SNR0', 'SNR6', 'SNR12', 'SNR18', 'SNR24']} 

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
    print("Clean signals loaded:")
    for key, value in clean_signals.items():
        print(f"{key}: length = {len(value)}")

    print("Noisy signals loaded:")
    for snr, signals in noisy_signals.items():
        for key, value in signals.items():
            print(f"{key} ({snr}): number of variants = {len(value)}")

    # Create the dataset and dataloader.
    segment_length = 1500  # 3 seconds at 500Hz
    dataset = ECGDataset(clean_signals, noisy_signals, segment_length)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)

    # Load in the COnvilutional Neural Network Class.
    net = CNN().float().to(device)
    
    # Set a default loss function (MSE)
    criterion = nn.MSELoss().to(device)
    
    # Use the RMSprop optimizer.
    optimizer = optim.RMSprop(net.parameters(), lr = 0.0002)

    # Training (Only train if specified.)
    if userSelectTrain:
        
        # Lets loop through the dataset 10 times (Epoch)
        for epoch in range(10):
            # Print Epoch Number.
            print("=========== EPOCH " + str(epoch) + " ===========")
            
            # Initialise a loss.
            running_loss = 0.0

            # Loop through the clean and corresponding noisy signals in the dataloader.
            for clean_signal, noisy_signal in dataloader:
                
                # Convert the clean and noisy signal to a float.
                clean_signal = clean_signal.float().to(device)  # Shape: (batch_size, 1, segment_length)
                noisy_signal = noisy_signal.float().to(device)  # Shape: (batch_size, 1, segment_length)

                # Reset all weights and biases to 0.
                optimizer.zero_grad()
                
                # Pass the noisy signal through our DNN.
                outputs = net(noisy_signal)
                
                # Calculate the peak locations.
                peaks = torch.tensor(detectors.hamilton_detector(clean_signal.cpu().numpy().flatten())).to(device)
                
                # Determine the loss (Using custom loss function)
                loss = lossfcn(clean_signal, outputs, peaks, a = 20)
                
                ### --------------Perform Back-Propagation ------------------------ ##
                
                # Compute the gradient of the loss with respect to all the parameters.
                loss.backward()
                
                # Update the parameters of the model based on the computed gradients.
                optimizer.step()
                
                ## ----------------- END Back-Propagation ------------------------ ##
                running_loss += loss.item()

            print(f'Epoch [{epoch + 1}/10], Loss: {running_loss / len(dataloader)}')

        print('Finished Training')

        # Save the model
        torch.save(net.state_dict(), './model_weightsCustom.pt')
        
    else:
        print('Using pre-trained model')
        
        # Load the saved model
        net.load_state_dict(torch.load('./model_weights.pt'))
        net.eval()

    # Evaluation - Plot only 5 signals
    with torch.no_grad():
        plot_count = 0
        for clean_signal, noisy_signal in dataloader:
            print(f'clean_signal type: {type(clean_signal)}, shape: {clean_signal.shape}')
            print(f'noisy_signal type: {type(noisy_signal)}, shape: {noisy_signal.shape}')

            clean_signal = clean_signal.float().to(device)  # Shape: (batch_size, 1, segment_length)
            noisy_signal = noisy_signal.float().to(device)  # Shape: (batch_size, 1, segment_length)

            outputs = net(noisy_signal)

            # Convert the tensors to numpy arrays for plotting
            noisy_signal_np = noisy_signal.cpu().numpy().flatten()
            clean_signal_np = clean_signal.cpu().numpy().flatten()
            outputs_np = outputs.cpu().numpy().flatten()

            plt.figure()
            plt.plot(noisy_signal_np, label='noisy')
            plt.plot(clean_signal_np, label='clean')
            plt.plot(outputs_np, label='denoised')
            plt.legend()
            plt.show()

            plot_count += 1
            if plot_count >= 5:
                break

    # Save the model
    torch.save(net.state_dict(), './model_weights.pt')
    print("Script execution completed.")
