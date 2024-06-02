class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(100, 128),
            nn.ReLU(True),
            nn.Linear(128, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, 100),  # Output size
            nn.Tanh())
    
    def forward(self, x):
        return self.main(x)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(100, 512),
            nn.ReLU(True),
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Linear(256, 1),
            nn.Sigmoid())
    
    def forward(self, x):
        return self.main(x)

generator = Generator()
discriminator = Discriminator()

criterion = nn.BCELoss()
optimizer_g = optim.Adam(generator.parameters(), lr=0.0002)
optimizer_d = optim.Adam(discriminator.parameters(), lr=0.0002)

num_epochs = 200
for epoch in range(num_epochs):
    for i, (data, _) in enumerate(train_loader):
        # Train Discriminator
        real_data = data
        batch_size = real_data.size(0)
        labels_real = torch.ones(batch_size, 1)
        labels_fake = torch.zeros(batch_size, 1)

        optimizer_d.zero_grad()
        outputs_real = discriminator(real_data)
        loss_real = criterion(outputs_real, labels_real)
        
        noise = torch.randn(batch_size, 100)
        fake_data = generator(noise)
        outputs_fake = discriminator(fake_data.detach())
        loss_fake = criterion(outputs_fake, labels_fake)
        
        loss_d = loss_real + loss_fake
        loss_d.backward()
        optimizer_d.step()

        # Train Generator
        optimizer_g.zero_grad()
        outputs = discriminator(fake_data)
        loss_g = criterion(outputs, labels_real)
        
        loss_g.backward()
        optimizer_g.step()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss D: {loss_d.item():.4f}, Loss G: {loss_g.item():.4f}")
