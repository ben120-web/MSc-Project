# module containing all models used in this experiment.
import torch
import torch.nn as nn

########################## Convolutional Neural Network Model definition #######################
class CNN(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        # First layer is a 1D conv layer.
        self.conv1 = nn.Conv1d(1, 16, 3, padding=1)  # 1 input channel, 16 output channels, kernel size 3
        self.conv2 = nn.Conv1d(16, 8, 3, padding=1)  # 16 input channels, 8 output channels, kernel size 3
        self.conv3 = nn.Conv1d(8, 1, 3, padding=1)   # 8 input channels, 1 output channel, kernel size 3
        # Removed the large kernel size in the final layer
        self.conv4 = nn.Conv1d(1, 1, 3, padding=1)   # 1 input channel, 1 output channel, kernel size 3
        self.acti = nn.ReLU()  # Activation function (ReLU)
        self.out = nn.Sigmoid()  # Sigmoid activation for the final output

    def forward(self, x):
        x = self.conv1(x)
        x = self.acti(x)
        x = self.conv2(x)
        x = self.acti(x)
        x = self.conv3(x)
        x = self.acti(x)
        x = self.conv4(x)
        out = self.out(x)  # Sigmoid activation applied directly
        return out  # No flattening, return the tensor directly

    
########################## Convolutional Denoising Auto-Encoder ##############################
class CDAE(nn.Module):
    
    # Initialization Function.
    def __init__(self):
        super().__init__()
        
        # Encoder part
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, 25, padding='same'),  # First Conv1D layer with 16 filters, kernel size 25
            nn.BatchNorm1d(16),                   # Batch Normalization
            nn.ReLU(),                            # Activation Function
            nn.MaxPool1d(2),                      # MaxPooling with size 2

            nn.Conv1d(16, 32, 25, padding='same'), # Second Conv1D layer with 32 filters, kernel size 25
            nn.BatchNorm1d(32),                   # Batch Normalization
            nn.ReLU(),                            # Activation Function
            nn.MaxPool1d(2),                      # MaxPooling with size 2

            nn.Conv1d(32, 64, 25, padding='same'), # Third Conv1D layer with 64 filters, kernel size 25
            nn.BatchNorm1d(64),                   # Batch Normalization
            nn.ReLU(),                            # Activation Function
            nn.MaxPool1d(2),                      # MaxPooling with size 2

            nn.Conv1d(64, 128, 25, padding='same'), # Fourth Conv1D layer with 128 filters, kernel size 25
            nn.BatchNorm1d(128),                  # Batch Normalization
            nn.ReLU(),                            # Activation Function
            nn.MaxPool1d(2),                      # MaxPooling with size 2

            nn.Conv1d(128, 1, 25, padding='same'), # Last Conv1D layer with 1 filter, kernel size 25
            nn.BatchNorm1d(1),                    # Batch Normalization
            nn.ReLU(),                            # Activation Function
        )
        
        # Decoder part
        self.decoder = nn.Sequential(
            nn.Conv1d(1, 128, 25, padding='same'), # First Conv1D layer in decoder with 128 filters
            nn.ReLU(),                            # Activation Function
            nn.Upsample(scale_factor=2),          # Up-sampling to double the size

            nn.Conv1d(128, 64, 25, padding='same'), # Second Conv1D layer in decoder with 64 filters
            nn.ReLU(),                            # Activation Function
            nn.Upsample(scale_factor=2),          # Up-sampling to double the size

            nn.Conv1d(64, 32, 25, padding='same'), # Third Conv1D layer in decoder with 32 filters
            nn.ReLU(),                            # Activation Function
            nn.Upsample(scale_factor=2),          # Up-sampling to double the size

            nn.Conv1d(32, 16, 25, padding='same'), # Fourth Conv1D layer in decoder with 16 filters
            nn.ReLU(),                            # Activation Function
            nn.Upsample(scale_factor=2),          # Up-sampling to double the size

            nn.Conv1d(16, 1, 25, padding='same'),  # Final Conv1D layer with 1 filter to reconstruct the signal
            nn.Sigmoid()                           # Sigmoid activation for output layer
        )

    # Set the forward pass.
    def forward(self, x):
        
        # Encode the input
        x = self.encoder(x)
        
        # Decode the encoded features
        x = self.decoder(x)
        
        return x
