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
    useCycleGAN = False # Set to true if you want to test cycle GAN.

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
    
    # load Cycle GAN if needed.
    if useCycleGAN:
        generator_A2B = Generator().float().to(device)
        generator_B2A = Generator().float().to(device)
        discriminator_A = Discriminator().float().to(device)
        discriminator_B = Discriminator().float().to(device)
        
        # Define optimisers for CycleGAN
        optimizer_G = optim.Adam(itertools.chain(generator_A2B.parameters(), generator_B2A.parameters(), lr = 0.0002, betas = (0.5, 0.999)))

        optimizer_D_A = optim.Adam(discriminator_A.parameters(), lr=0.0002, betas=(0.5, 0.999))
        
        optimizer_D_B = optim.Adam(discriminator_B.parameters(), lr=0.0002, betas=(0.5, 0.999))

    # Set the loss function and optimizer
    criterion = torch.nn.MSELoss().to(device)
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
    
    def adversarial_loss_D(real, fake):
        """
        Adversarial loss for the discriminator.
        """
        return nn.BCELoss()(real, torch.ones_like(real)) + nn.BCELoss()(fake, torch.zeros_like(fake))

    def adversarial_loss_G(fake):
        """
        Adversarial loss for the generator.
        """
        return nn.BCELoss()(fake, torch.ones_like(fake)) 
        
    def cycle_consistency_loss(real_X, reconstructed_X, real_Y, reconstructed_Y):
        """
        Cycle consistency loss.
        """
        return nn.L1Loss()(reconstructed_X, real_X) + nn.L1Loss()(reconstructed_Y, real_Y)
        
    def identity_loss(real_X, same_X, real_Y, same_Y):
        """
        Identity loss.
        """
        return nn.L1Loss()(same_X, real_X) + nn.L1Loss()(same_Y, real_Y)

    def distance_loss(GX_to_Y_output, real_Y, GY_to_X_output, real_X):
        """
        Distance loss (ldist).
        """
        return nn.L1Loss()(GX_to_Y_output, real_Y) + nn.L1Loss()(GY_to_X_output, real_X)
    
    def max_differential_loss(GX_to_Y_output, real_Y):
        """
        Maximum differential loss (lmax).
        """
        # Here, we'll use max pooling to simulate the "max differential" idea from the paper.
        # This can be a custom operation depending on how the paper defines "max differential".
        return torch.mean(torch.abs(torch.max(GX_to_Y_output) - torch.max(real_Y)))
    
    def total_loss(GX_to_Y_output, real_Y, GY_to_X_output, real_X, 
                fake_X, fake_Y, reconstructed_X, reconstructed_Y, 
                same_X, same_Y, DY, DX, 
                lambda_cyc=10, lambda_id=1, lambda_dist=10, lambda_max=1):
        """
        Total loss function (Losstotal) as described in the paper.
        """
        # Adversarial losses
        loss_adv1 = adversarial_loss_G(DY(GX_to_Y_output))
        loss_adv2 = adversarial_loss_G(DX(GY_to_X_output))

        # Cycle consistency loss
        loss_cyc = cycle_consistency_loss(real_X, reconstructed_X, real_Y, reconstructed_Y)

        # Identity loss
        loss_id = identity_loss(real_X, same_X, real_Y, same_Y)

        # Distance loss
        loss_dist = distance_loss(GX_to_Y_output, real_Y, GY_to_X_output, real_X)

        # Maximum differential loss
        loss_max = max_differential_loss(GX_to_Y_output, real_Y)

        # Total loss
        total_loss = loss_adv1 + loss_adv2 + lambda_cyc * loss_cyc + lambda_id * loss_id + lambda_dist * loss_dist + lambda_max * loss_max
        
        return total_loss
    
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
        if test_signal.shape != reference_signal.shape:
            raise ValueError("The two signals must have the same length.")
        
        # Calculate the RMSE
        rmse = np.sqrt(np.mean((test_signal - reference_signal) ** 2))
        
        return rmse

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
                    #loss = criterion
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()

            print(f'Epoch [{epoch + 1}/10], Loss: {running_loss / len(dataloader)}')
        torch.save(net.state_dict(), './model_weightsCDAE12dB.pt')

    else:
        net.load_state_dict(torch.load('./model_weightsCustom.pt'))
        net.eval()

    # Initialise.
    rmse_noisy = []
    rmse_processed = []
    
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
    
if __name__ == '__main__':
    main()
