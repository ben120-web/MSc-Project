import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(100, 50),  # Adjust input size as necessary
            nn.ReLU(True),
            nn.Linear(50, 20),
            nn.ReLU(True))
        self.decoder = nn.Sequential(
            nn.Linear(20, 50),
            nn.ReLU(True),
            nn.Linear(50, 100),
            nn.ReLU(True))
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

model = AutoEncoder()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32))
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

num_epochs = 20
for epoch in range(num_epochs):
    for data in train_loader:
        inputs = data[0]
        outputs = model(inputs)
        loss = criterion(outputs, inputs)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# Evaluate the model
with torch.no_grad():
    reconstructed = model(torch.tensor(X_test, dtype=torch.float32))
    test_loss = criterion(reconstructed, torch.tensor(X_test, dtype=torch.float32))
    print(f"Test Loss: {test_loss.item():.4f}")
