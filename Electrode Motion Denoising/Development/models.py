# module containing all models used in this experiment.
import torch
import torch.nn as nn
import torch.nn.functional as F

########################## Convolutional Neural Network Model definition #######################
class CNN(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        # First layer is a 1D conv layer.
        self.conv1 = nn.Conv1d(1, 16, 3, padding='same')  # 1 input channel, 16 output channels, kernel size 3
        self.conv2 = nn.Conv1d(16, 8, 3, padding='same')  # 16 input channels, 8 output channels, kernel size 3
        self.conv3 = nn.Conv1d(8, 1, 3, padding='same')   # 8 input channels, 1 output channel, kernel size 3
        # Removed the large kernel size in the final layer
        self.conv4 = nn.Conv1d(1, 1, 1500, padding='same')   # 1 input channel, 1 output channel, kernel size 1500 (Same as input length)
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
        x = torch.flatten(x, 1) # Flatten all dimensions except batch.
        out = self.out(x)  # Sigmoid activation applied directly
        return out  # No flattening, return the tensor directly

    
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
      
      
######################### ECA-Net and CycleGAN #################################
class ECALayer(nn.Module):
  
  def __init__(self, channel, k_szie = 3):
    
    super(ECALayer, self).__init__()
    self.avg_pool = nn.AdaptiveAvgPool1d(1)
    self.conv = nn.Conv1d(1, 1, kernel_size = k_size, padding = (k_size - 1) // 2, bias = False)
    self.sigmoid = nn.Sigmoid()
    
    
  def forward(self, x):
    
    y = self.avg_pool(x)
    y = self.conv(y.transpose(-1, -2)).transpose(-1, -2)
    y = self.sigmoid(y)
    
    return x * y.expand_as(x)
  
class SelfONN(nn.Module):
    
  def __init__(self, in_channels, out_channels, kernel_size = 5, stride = 1, padding = 'same'):
    
    super(SelfONN, self).__init__()
    
    self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding=padding)
    
    self.bn = nn.BatchNorm1d(out_channels)
    
    self.relu = nn.ReLU()
    
  def forward(self, x ):
    
    x = self.conv(x)
    x = self.bn(x)
    x = self.relu(x)
    
    return x
    
class GeneratorUNet(nn.Module):
  
    def __init__(self):
        super(GeneratorUNet, self).__init__()
        
        self.down1 = SelfONN(1, 16)
        self.down2 = SelfONN(16, 32)
        self.down3 = SelfONN(32, 64)
        self.down4 = SelfONN(64, 128)
        self.down5 = SelfONN(128, 128)
        
        self.up1 = nn.ConvTranspose1d(128, 128, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.up2 = nn.ConvTranspose1d(128, 64, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.up3 = nn.ConvTranspose1d(64, 32, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.up4 = nn.ConvTranspose1d(32, 16, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.up5 = nn.ConvTranspose1d(16, 1, kernel_size=6, stride=2, padding=2, output_padding=1)
        
        # ECA Layers after each upsampling
        self.eca1 = ECALayer(128)
        self.eca2 = ECALayer(64)
        self.eca3 = ECALayer(32)
        self.eca4 = ECALayer(16)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.final = nn.Sigmoid()
    
    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        
        u1 = self.up1(d5)
        u1 = self.eca1(u1)
        u2 = self.up2(u1)
        u2 = self.eca2(u2)
        u3 = self.up3(u2)
        u3 = self.eca3(u3)
        u4 = self.up4(u3)
        u4 = self.eca4(u4)
        u5 = self.up5(u4)
        
        return self.final(u5)
      
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.layer1 = SelfONN(1, 128)
        self.layer2 = SelfONN(128, 128)
        self.layer3 = SelfONN(128, 256)
        self.layer4 = SelfONN(256, 256)
        self.layer5 = nn.Conv1d(256, 1, kernel_size=5, padding='same')
        
        self.final = nn.Sigmoid()
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        
        return self.final(x)

      
class CycleGAN(nn.Module):
    def __init__(self):
        super(CycleGAN, self).__init__()
        self.G_A2B = GeneratorUNet()
        self.G_B2A = GeneratorUNet()
        self.D_A = Discriminator()
        self.D_B = Discriminator()
    
    def forward(self, x):
        # Generator A to B
        fake_B = self.G_A2B(x)
        recon_A = self.G_B2A(fake_B)
        
        # Generator B to A
        fake_A = self.G_B2A(x)
        recon_B = self.G_A2B(fake_A)
        
        return fake_A, fake_B, recon_A, recon_B    
