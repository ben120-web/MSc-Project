# module containing all models used in this experiment.
import torch
import torch.nn as nn
import torch.nn.functional as F
    
########################## Convolutional Denoising Auto-Encoder ##############################
class CDAE(nn.Module):
    
    def __init__(self):
        super(CDAE, self).__init__()
        
        # Encoder part
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, 25, stride=2, padding=12),  # Downsample by 2
            nn.BatchNorm1d(16),                         
            nn.ReLU(),                                  

            nn.Conv1d(16, 32, 25, stride=2, padding=12),  # Downsample by 2
            nn.BatchNorm1d(32),                         
            nn.ReLU(),                                  

            nn.Conv1d(32, 64, 25, stride=2, padding=12),  # Downsample by 2
            nn.BatchNorm1d(64),                         
            nn.ReLU(),                                  

            nn.Conv1d(64, 128, 25, stride=2, padding=12),  # Downsample by 2
            nn.BatchNorm1d(128),                        
            nn.ReLU(),                                  

            nn.Conv1d(128, 256, 25, stride=2, padding=12),  # Downsample by 2
            nn.BatchNorm1d(256),                         
            nn.ReLU()                                   
        )
        
        # Decoder part
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(256, 128, 25, stride=2, padding=12, output_padding=1),  # Upsample by 2
            nn.ReLU(),                                            

            nn.ConvTranspose1d(128, 64, 25, stride=2, padding=12, output_padding=1),  # Upsample by 2
            nn.ReLU(),                                            

            nn.ConvTranspose1d(64, 32, 25, stride=2, padding=12, output_padding=1),   # Upsample by 2
            nn.ReLU(),                                            

            nn.ConvTranspose1d(32, 16, 25, stride=2, padding=12, output_padding=1),   # Upsample by 2
            nn.ReLU(),                                            

            nn.ConvTranspose1d(16, 1, 25, stride=2, padding=12, output_padding=1),    # Final ConvTranspose1d layer
            nn.Sigmoid()                                           
        )

    def forward(self, x):
        # Encode the input
        x = self.encoder(x)
        
        # Decode the encoded features
        x = self.decoder(x)
        
        return x


########################## Region Based Convolutional Neural Network ##################################
class RCNN(nn.Module):
    def __init__(self, input_size):
        super(RCNN, self).__init__()
    
        self.input_size = input_size

        self.cv1_k = 3
        self.cv1_s = 1
        self.cv1_out = int(((self.input_size - self.cv1_k)/self.cv1_s) + 1)

        self.cv2_k = 3
        self.cv2_s = 1
        self.cv2_out = int(((self.cv1_out - self.cv2_k)/self.cv2_s) + 1)

        self.cv3_k = 5
        self.cv3_s = 1
        self.cv3_out = int(((self.cv2_out - self.cv3_k)/self.cv3_s) + 1)

        self.cv4_k = 5
        self.cv4_s = 1
        self.cv4_out = int(((self.cv3_out - self.cv4_k)/self.cv4_s) + 1)
    
    
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
          nn.Linear(self.cv4_out, 1500), # FC Layer
          nn.Linear(1500, 1500) # Regression
        )
        
    def forward(self, x):
        x = self.layer_1(x) 
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)
        x = x.view(x.size(0), -1)
        x = self.layer_5(x)
    
        return x
      

class DRNN(nn.Module):
  
    def __init__(self, input_size=1, lstm_hidden_size=64, fully_connected_size=64, output_size=1, seq_length=1500):
        super(DRNN, self).__init__()
        
        # LSTM layer: input_size = 1 (assuming time series input of 1D signal), hidden size = 64 units
        self.lstm = nn.LSTM(input_size, lstm_hidden_size, batch_first=True)
        
        # Fully connected layers
        self.fc1 = nn.Linear(lstm_hidden_size * seq_length, fully_connected_size)  # Multiplied by sequence length for FC
        self.fc2 = nn.Linear(fully_connected_size, fully_connected_size)
        
        # Output layer (1D regression task)
        self.output = nn.Linear(fully_connected_size, output_size)
        
        # Activation function
        self.relu = nn.ReLU()

    def forward(self, x):
        # LSTM forward pass
        lstm_out, (hn, cn) = self.lstm(x)
        
        # Flatten the LSTM output
        lstm_out = lstm_out.contiguous().view(x.size(0), -1)  # Flatten to feed into fully connected layers
        
        # Fully connected layers with ReLU
        x = self.fc1(lstm_out)
        x = self.relu(x)
        
        x = self.fc2(x)
        x = self.relu(x)
        
        # Output layer
        out = self.output(x)
        return out