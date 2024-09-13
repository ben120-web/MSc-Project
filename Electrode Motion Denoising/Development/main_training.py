import torch
import torch.nn as nn
import torch.optim as optim
from ecgdetectors import Detectors
from data_loading import load_data
from models import RCNN
import math
import matplotlib.pyplot as plt
import numpy as np
from models import CDAE

def main():
    # Set user selection
    userSelectTrain = True
    
    # Define the device to use
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('Using: ' + str(torch.cuda.get_device_name(device)))
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print('Using: MPS')
    else:
        device = torch.device('cpu')
        print('Using: CPU')

    # Set the Peak detectors to work at 500Hz
    detectors = Detectors(500)

    # Define paths
    clean_signals_path = '/Users/benrussell/Library/CloudStorage/GoogleDrive-rben3625@gmail.com/My Drive/cleanSignals/'
    noisy_signals_path = '/Users/benrussell/Library/CloudStorage/GoogleDrive-rben3625@gmail.com/My Drive/noisySignals/'

    # Load the data
    dataloader = load_data(clean_signals_path, noisy_signals_path, segment_length = 1500, batch_size=1, num_workers=0)

    # Load the model
    net = CDAE().float().to(device)

    # Set the loss function and optimizer
    optimizer = optim.RMSprop(net.parameters(), lr=0.002)

    ##################### HELPER FUNCTIONS ###########################
    def lossfcn(y_true, y_pred, peaks, a=20):
        criterion = torch.nn.MSELoss().to(device)
        alpha = a
        loss = 0.0
        R = 0.0
        for x, y, z in zip(y_pred, y_true, peaks):
            qrs_loss = []
            if z.dim() > 0:
                z = z[z.nonzero(as_tuple=True)].squeeze()
                for qrs in z:
                    max_ind = qrs.item() + 1
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
                
        # If y_pred is 1D after squeezing
        if y_pred.dim() == 1:
            y_pred = y_pred[:1500]
        else:
            y_pred = y_pred[:, :1500]  # Assuming it has more dimensions

        # Now compute the loss
        loss = criterion(y_true, y_pred) + torch.mean(torch.tensor(R))
        (torch.tensor(R))
        return loss
    
    def calculate_rmse(test_signal, reference_signal):
        """
        Calculate the Root Mean Square Error (RMSE) between two time series signals.

        Parameters:
        signal1 (array-like): The first time series signal.
        signal2 (array-like): The second time series signal.

        Returns:
        float: The RMSE between the two signals.
        """
        # Convert the signals to numpy arrays in case they are not already
        signal1 = np.array(test_signal)
        signal2 = np.array(reference_signal)
        
        # Ensure the two signals have the same length
        if signal1.shape != signal2.shape:
            
            signal1
            
            raise ValueError("The two signals must have the same length.")
        
        # Calculate the RMSE
        rmse = np.sqrt(np.mean((signal1 - signal2) ** 2))
        
        return rmse
    
    def normalized_cross_correlation(signal1, signal2):
        """
        Calculate the normalized cross-correlation between two time series signals.

        Parameters:
        signal1 (array-like): First time series signal.
        signal2 (array-like): Second time series signal.

        Returns:
        ncc (array): Normalized cross-correlation of the two signals.
        """
        # Ensure the signals are numpy arrays
        signal1 = np.asarray(signal1)
        signal2 = np.asarray(signal2)

        # Check if the signals have the same length
        if signal1.shape != signal2.shape:
            raise ValueError("The two signals must have the same length.")
        
        # Subtract the mean from the signals (zero-mean)
        signal1_zero_mean = signal1 - np.mean(signal1)
        signal2_zero_mean = signal2 - np.mean(signal2)

        # Compute the cross-correlation
        cross_corr = np.correlate(signal1_zero_mean, signal2_zero_mean, mode='full')

        # Normalize the cross-correlation
        normalization_factor = np.std(signal1) * np.std(signal2) * len(signal1)
        ncc = cross_corr / normalization_factor

        return ncc 
    
    def calculate_snr(signal, noise):
        """
        Calculate the Signal-to-Noise Ratio (SNR).
        
        SNR = 10 * log10(P_signal / P_noise)
        
        Parameters:
        signal (array-like): The original signal.
        noise (array-like): The noise in the signal.
        
        Returns:
        float: The SNR value in dB.
        """
        # Calculate the power of the signal and noise
        signal_power = np.mean(signal ** 2)
        noise_power = np.mean(noise ** 2)
        
        # Calculate the SNR in decibels (dB)
        snr = 10 * np.log10(signal_power / noise_power)
        
        return snr

    def snr_improvement(clean_signal, noisy_signal, processed_signal):
        """
        Calculate the SNR improvement after signal processing.

        Parameters:
        clean_signal (array-like): The clean, reference signal.
        noisy_signal (array-like): The noisy input signal.
        processed_signal (array-like): The signal after processing (denoised signal).
        
        Returns:
        float: The SNR improvement in dB.
        """
        # Calculate the noise before and after processing
        noise_before = noisy_signal - clean_signal
        noise_after = processed_signal - clean_signal
        
        # Calculate SNR before and after processing
        snr_before = calculate_snr(clean_signal, noise_before)
        snr_after = calculate_snr(clean_signal, noise_after)
        
        # Calculate SNR improvement
        snr_improvement = snr_after - snr_before
        
        return snr_improvement

    ############################## Training ############################
    if userSelectTrain:
        for epoch in range(10):
            print(f"=========== EPOCH {epoch} ===========")
            running_loss = 0.0
            
            for batch_idx, (clean_signals, noisy_signals) in enumerate(dataloader):
                print(f"Batch {batch_idx}: clean_signal shape = {clean_signals.shape}, noisy_signal shape = {noisy_signals.shape}")
                
                # Continue with the rest of the processing
                clean_signal = clean_signals.float().to(device)
                noisy_signal = noisy_signals.float().to(device)
                
                for i in range(noisy_signal.size(1)):
                    noisy_segment = noisy_signal[:, i, :]
                    optimizer.zero_grad()
                    outputs = net(noisy_segment)
                    peaks = torch.tensor(detectors.hamilton_detector(clean_signal.cpu().numpy().flatten())).to(device)
                    loss = lossfcn(clean_signal.squeeze(), outputs.squeeze(), peaks, a=20)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()

            print(f'Epoch [{epoch + 1}/10], Loss: {running_loss / len(dataloader)}')
        torch.save(net.state_dict(), './model_weightsCDAE0dB.pt')

    else:
        net.load_state_dict(torch.load('./model_weightsRCNN24dB.pt'))
        net.eval()

    # Initialise.
    rmse_noisy = []
    rmse_processed = []
    ncc_noisy = []
    ncc_processed = []
    snr_impove = []
    
    # Evaluation
    with torch.no_grad():
        plot_count = 0
        
        for clean_signal, noisy_signal in dataloader:
            
            clean_signal = clean_signal.float().to(device)
            noisy_signal = noisy_signal.float().to(device)
            
            for i in range(noisy_signal.size(1)):
                noisy_segment = noisy_signal[:, i, :, :]
                outputs = net(noisy_segment)
                noisy_signal_np = noisy_segment.cpu().numpy().flatten()
                clean_signal_np = clean_signal.cpu().numpy().flatten()
                outputs_np = outputs.cpu().numpy().flatten()
                
                # Calculate the RMSE between clean and noisy.
                rmse_clean_vs_noisy = calculate_rmse(noisy_signal_np, clean_signal_np)
                
                # Append to a list.
                rmse_noisy.append(rmse_clean_vs_noisy)
                
                # Calculate the RMSE between the clean and processed.
                rmse_clean_vs_processed = calculate_rmse(outputs_np, clean_signal_np)
                
                # Append to list.
                rmse_processed.append(rmse_clean_vs_processed)
                
                # Calculate the NCC between clean and noisy.
                ncc_clean_noisy = normalized_cross_correlation(noisy_signal_np, clean_signal_np)
                
                # Append to list.
                ncc_noisy.append(ncc_clean_noisy)
                
                # Calculate NCC between clean and processed.
                ncc_clean_processed = normalized_cross_correlation(outputs_np, clean_signal_np)
                
                # Append to list.
                ncc_processed.append(ncc_clean_processed)
                
                # Deteremine the SNR improvement
                snr_improvement_val = snr_improvement(clean_signal_np, noisy_signal_np, outputs_np)
                
                # Append to list.
                snr_impove.append(snr_improvement_val)
                
                # Plot examples.
                plt.figure()
                plt.plot(noisy_signal_np, label='noisy')
                plt.plot(clean_signal_np, label='clean')
                plt.plot(outputs_np, label='denoised')
                plt.legend()
                plt.show()
                plot_count += 1
                if plot_count >= 5:
                    break

    torch.save(net.state_dict(), './model_weights.pt')
    print("Script execution completed.")

    # Calculate the mean RMSE
    mean_rmse_noisy = np.mean(rmse_noisy)
    mean_rmse_processed = np.mean(rmse_processed)
    
    # Print results.
    print('mean RMSE for noisy signals : ' + str(mean_rmse_noisy))
    print('mean RMSE for noisy signals : ' + str(mean_rmse_processed))
    
    mean_ncc_noisy = np.mean(ncc_noisy)
    mean_ncc_processed = np.mean(ncc_processed)
    
    # Print results.
    print('mean NCC for noisy signals : ' + str(mean_ncc_noisy))
    print('mean NCC for noisy signals : ' + str(mean_ncc_processed))
    
    mean_snr_improvement = np.mean(snr_impove)
    
    print('mean SNR improvement for signals : ' + str(mean_snr_improvement))
    
if __name__ == '__main__':
    main()
