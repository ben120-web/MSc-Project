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

# Let's define a device to use. (GPU if desktop being used, MPS if Mac Book being used, CPU otherwise.)
if torch.cuda.is_available():
    device = torch.device('cuda')
    print('Using: ' + str(torch.cuda.get_device_name(device)))
elif torch.backends.mps.is_available():
    device = torch.device('mps')
    print('Using: MPS')
else:
    device = torch.device('cpu')
    print('Using: CPU')

# Set the Peak detectors to work at 500Hz.
detectors = Detectors(500)

# Set path to Google Drive. (This needs mounted to your machine.)
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

# Custom Loss Function (This calculates the Mean Square Error around Heartbeats.)
def lossfcn(y_true, y_pred, peaks, a=20):
    # Set the Criterion to use Mean Square Error.
    criterion = nn.MSELoss().to(device)
    alpha = a
    loss = 0.0
    R = 0.0

    # Loop through the predicted denoised signal, the correct clean signal and the peak locations.
    for x, y, z in zip(y_pred, y_true, peaks):
        # Initialize the loss around the QRS.
        qrs_loss = []

        # Ensure z is a tensor and handle indices properly
        if z.dim() > 0:
            z = z[z.nonzero(as_tuple=True)].squeeze()

            # Loop through the QRS locations (Peaks)
            for qrs in z:
                # Get the last peak in the signal.
                max_ind = qrs.item() + 1

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

# Dataset class definition
class ECGDataset(Dataset):
    def __init__(self, clean_signals, noisy_signals, segment_length=1500):
        self.clean_signals = clean_signals
        self.noisy_signals = noisy_signals
        self.signal_numbers = list(clean_signals.keys())
        self.segment_length = segment_length

    def __len__(self):
        # Calculate the total number of segments in the entire dataset
        total_segments = 0
        for signal_number in self.signal_numbers:
            signal_length = self.clean_signals[signal_number].size
            segments_per_signal = signal_length // self.segment_length
            total_segments += segments_per_signal * len(self.noisy_signals)  # Multiply by the number of SNR levels

        return total_segments

    def __getitem__(self, idx):
        # Calculate the total number of segments per signal (across all SNRs)
        segments_per_signal = 10  # Assuming 15000 samples per signal and 1500 samples per segment

        # Determine the signal index based on idx
        signal_idx = idx // (segments_per_signal * len(self.noisy_signals))

        # Ensure that signal_idx is within range
        if signal_idx >= len(self.signal_numbers):
            raise IndexError(f"Signal index {signal_idx} out of range for available signals")

        # Determine the segment index within that signal and SNR
        segment_in_signal_idx = idx % (segments_per_signal * len(self.noisy_signals))
        snr_idx = segment_in_signal_idx // segments_per_signal
        segment_idx = segment_in_signal_idx % segments_per_signal

        # Get the signal number based on the signal index
        signal_number = self.signal_numbers[signal_idx]
        snr = list(self.noisy_signals.keys())[snr_idx]

        # Calculate the start and end indices for the segment
        start_idx = segment_idx * self.segment_length
        end_idx = start_idx + self.segment_length

        # Check if the signal number exists in the noisy signals for the current SNR
        if signal_number in self.noisy_signals[snr]:
            # Access the clean signal array and extract the corresponding segment
            clean_signal_data = self.clean_signals[signal_number][0]  # Access the entire signal
            clean_signal_segment = clean_signal_data[start_idx:end_idx]

            # Prepare a list to store all noisy signal segments for this clean segment
            noisy_signal_segments = []

            # Extract the corresponding noisy signal segment for each copy
            for noisy_signal_copy in self.noisy_signals[snr][signal_number]:
                noisy_signal_segment = noisy_signal_copy[0][start_idx:end_idx]
                noisy_signal_segments.append(noisy_signal_segment)

            # Convert the clean segment to a tensor
            clean_signal_tensor = torch.tensor(clean_signal_segment).unsqueeze(0)  # Shape: [1, 1500]

            # Stack all noisy segments into a tensor
            noisy_signal_tensor = torch.tensor(noisy_signal_segments)  # Shape: [num_noisy_copies, 1500]

            # Add a channel dimension to both clean and noisy signals
            clean_signal_tensor = clean_signal_tensor.unsqueeze(0)  # Shape: [1, 1, 1500]
            noisy_signal_tensor = noisy_signal_tensor.unsqueeze(1)  # Shape: [num_noisy_copies, 1, 1500]

            return clean_signal_tensor, noisy_signal_tensor

        # If no corresponding noisy signal exists, return an empty tensor pair
        return torch.tensor([]), torch.tensor([])

# Convolutional Neural Network Model definition
class CNN(nn.Module):
    
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

# Start processing (Data Loading, Training, Evaluation)
if __name__ == '__main__':
    print("Starting script execution...")

    # Load clean signals
    clean_signals = {}
    
    # Loop through each h5 file in the directory.
    for file in glob.glob(os.path.join(clean_signals_path, '*.h5')):
        
        # Get the signal number. 
        signal_number = os.path.basename(file).split('_')[1]
        
        # Load in all h5 files based on signal number.
        clean_signals[signal_number] = load_h5_file(file)

    # Load noisy signals and group by signal number
    noisy_signals = {snr: {} for snr in ['SNR0', 'SNR12', 'SNR18', 'SNR24']} 

    # Loop through each SNR level.
    for snr in noisy_signals.keys():
        
        # Set the path to the specific SNR under test.
        snr_path = os.path.join(noisy_signals_path, snr)

        # Loop through each file in the directory of the SNR.
        for file in glob.glob(os.path.join(snr_path, '*.h5')):
            
            # Extract the signal number from the file name.
            signal_number = os.path.basename(file).split('-')[0].split('_')[1]

            # If the signal number is not in noisy signals, set it as empty.
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

    # Load in the Convolutional Neural Network Class.
    net = CNN().float().to(device)
    
    # Set a default loss function (MSE)
    criterion = nn.MSELoss().to(device)
    
    # Use the RMSprop optimizer.
    optimizer = optim.RMSprop(net.parameters(), lr=0.0002)

    # Training (Only train if specified.)
    if userSelectTrain:
        
        # Let's loop through the dataset 10 times (Epoch)
        for epoch in range(10):
            # Print Epoch Number.
            print("=========== EPOCH " + str(epoch) + " ===========")
            
            # Initialize a loss.
            running_loss = 0.0

            # Loop through the clean and corresponding noisy signals in the dataloader.
            for clean_signal, noisy_signal in dataloader:
                
                # Convert the clean and noisy signal to a float.
                clean_signal = clean_signal.float().to(device)  # Shape: (batch_size, 1, 1, 1500)
                noisy_signal = noisy_signal.float().to(device)  # Shape: (batch_size, 110, 1, 1500)
                
                # Iterate over each noisy segment.
                for i in range(noisy_signal.size(1)):  # Iterate over the 110 noisy segments
                    noisy_segment = noisy_signal[:, i, :, :]  # Shape: (batch_size, 1, 1500)

                    # Reset all weights and biases to 0.
                    optimizer.zero_grad()
                    
                    # Pass the noisy signal through our DNN.
                    outputs = net(noisy_segment)
                    
                    # Reshape the outputs back to match the clean signal shape for loss calculation.
                    outputs = outputs.view(-1, 1, 1500)  # Shape: (batch_size, 1, 1500)

                    # Calculate the peak locations.
                    peaks = torch.tensor(detectors.hamilton_detector(clean_signal.cpu().numpy().flatten())).to(device)
                    
                    # Determine the loss (Using custom loss function)
                    loss = lossfcn(clean_signal.squeeze(), outputs.squeeze(), peaks, a=20)
                    
                    # Compute the gradient of the loss with respect to all the parameters.
                    loss.backward()
                    
                    # Update the parameters of the model based on the computed gradients.
                    optimizer.step()
                    
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
    with torch.no_grad():  # No updates to weights and biases. Uses trained values.
        
        # Initialize a counter for plots.
        plot_count = 0
        
        # Loop through each clean and noisy signal.
        for clean_signal, noisy_signal in dataloader:
            
            # Print the type and shape of data.
            print(f'clean_signal type: {type(clean_signal)}, shape: {clean_signal.shape}')
            print(f'noisy_signal type: {type(noisy_signal)}, shape: {noisy_signal.shape}')

            # Convert signals to float tensors.
            clean_signal = clean_signal.float().to(device)  # Shape: (batch_size, 1, 1, 1500)
            noisy_signal = noisy_signal.float().to(device)  # Shape: (batch_size, 110, 1, 1500)

            # Iterate over each noisy segment.
            for i in range(noisy_signal.size(1)):  # Iterate over the 110 noisy segments
                noisy_segment = noisy_signal[:, i, :, :]  # Shape: (batch_size, 1, 1500)

                # Pass the noisy signal through the network.
                outputs = net(noisy_segment)

                # Convert the tensors to numpy arrays for plotting.
                noisy_signal_np = noisy_segment.cpu().numpy().flatten()
                clean_signal_np = clean_signal.cpu().numpy().flatten()
                outputs_np = outputs.cpu().numpy().flatten()

                # Plot the signals.
                plt.figure()
                plt.plot(noisy_signal_np, label='noisy')
                plt.plot(clean_signal_np, label='clean')
                plt.plot(outputs_np, label='denoised')
                plt.legend()
                plt.show()

                plot_count += 1
                
                # Only plot 5 signals.
                if plot_count >= 5:
                    break

    # Save the model
    torch.save(net.state_dict(), './model_weights.pt')
    print("Script execution completed.")
