# train_vae.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torchvision import utils
from data_loader import get_dataloaders
from utils import save_image_batch
from logger import TrainingLogger
from checkpoint import save_checkpoint
import os
import csv

def main():
    # ------------------ 1. Configuration ------------------
    batch_size = 128
    epochs = 10
    lr = 1e-3
    latent_dim = 20
    checkpoint_interval = 2
    device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs("outputs/vae_outputs", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    # ------------------ 2. Dataset ------------------
    train_loader, test_loader = get_dataloaders(batch_size=batch_size, augment=True)

    # ------------------ 3. Model Definition ------------------
    class VAE(nn.Module):
        def __init__(self, latent_dim=20):
            super(VAE, self).__init__()
            self.fc1 = nn.Linear(28 * 28, 400)
            self.fc21 = nn.Linear(400, latent_dim)
            self.fc22 = nn.Linear(400, latent_dim)
            self.fc3 = nn.Linear(latent_dim, 400)
            self.fc4 = nn.Linear(400, 28 * 28)

        def encode(self, x):
            h1 = F.relu(self.fc1(x))
            return self.fc21(h1), self.fc22(h1)

        def reparameterize(self, mu, logvar):
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std

        def decode(self, z):
            h3 = F.relu(self.fc3(z))
            return torch.sigmoid(self.fc4(h3))

        def forward(self, x):
            mu, logvar = self.encode(x.view(-1, 28 * 28))
            z = self.reparameterize(mu, logvar)
            recon = self.decode(z)
            return recon, mu, logvar

    def loss_function(recon_x, x, mu, logvar):
        BCE = F.binary_cross_entropy(recon_x, x.view(-1, 28 * 28), reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KLD

    model = VAE(latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    logger = TrainingLogger(folder="logs")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0

        for data, _ in train_loader:
            data = data.to(device)
            optimizer.zero_grad()

            recon_batch, mu, logvar = model(data)
            loss = loss_function(recon_batch, data, mu, logvar)
            loss.backward()
            total_loss += loss.item()
            optimizer.step()

        avg_loss = total_loss / len(train_loader.dataset)
        print(f"Epoch {epoch}/{epochs} | Average Loss: {avg_loss:.4f}")
        logger.log(epoch, avg_loss)

        if epoch % checkpoint_interval == 0 or epoch == epochs:
            save_checkpoint(model, optimizer, epoch, avg_loss, folder="checkpoints", prefix="vae")

        with torch.no_grad():
            z = torch.randn(64, latent_dim).to(device)
            samples = model.decode(z).cpu()
            save_image_batch(samples.view(64, 1, 28, 28), filename=f"sample_epoch_{epoch}.png", folder="outputs/vae_outputs")

            data, _ = next(iter(test_loader))
            data = data.to(device)
            recon, _, _ = model(data)
            recon = recon.view(-1, 1, 28, 28)
            comparison = torch.cat([data[:8], recon[:8]])
            utils.save_image(comparison.cpu(), f"outputs/vae_outputs/reconstruction_epoch_{epoch}.png", nrow=8)

    # ------------------ 4. Save final result to CSV ------------------
    csv_path = "results/vae_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "Epochs", "Final_Loss"])
        writer.writerow(["VAE", epochs, round(avg_loss, 4)])

    print(f"✅ Saved final VAE results to {csv_path}")
    print("Training complete. Check 'outputs/vae_outputs/' for images and 'logs/' for metrics.")


if __name__ == "__main__":
    main()
