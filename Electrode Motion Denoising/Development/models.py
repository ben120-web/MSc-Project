# module containing all models used in this experiment.
import torch
import torch.nn as nn

########################## Convolutional Neural Network Model definition #######################
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
