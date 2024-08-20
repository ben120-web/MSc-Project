import torch
import torch.optim as optim
from ecgdetectors import Detectors
from data_loading import load_data
from models import CNN
import math
import matplotlib.pyplot as plt

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
    net = CNN().float().to(device)

    # Set the loss function and optimizer
    criterion = torch.nn.MSELoss().to(device)
    optimizer = optim.RMSprop(net.parameters(), lr=0.0002)

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
        loss = criterion(y_true, y_pred) + torch.mean(torch.tensor(R))
        return loss

    # Training
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
        torch.save(net.state_dict(), './model_weightsCustom.pt')

    else:
        net.load_state_dict(torch.load('./model_weightsCustom.pt'))
        net.eval()

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

if __name__ == '__main__':
    main()
