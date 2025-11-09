# train_vqvae.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import utils
from data_loader import get_dataloaders
from utils import save_image_batch
from logger import TrainingLogger
from checkpoint import save_checkpoint
import os
import csv


# ------------------ 1. Vector Quantizer ------------------
class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings=128, embedding_dim=64, commitment_cost=0.25):
        super(VectorQuantizer, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1 / num_embeddings, 1 / num_embeddings)

    def forward(self, z):
        # z: [B, C, H, W]
        z_flattened = z.permute(0, 2, 3, 1).contiguous()  # [B, H, W, C]
        z_flattened = z_flattened.view(-1, self.embedding_dim)

        # Compute distances between z and embedding vectors
        distances = (
            torch.sum(z_flattened**2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight**2, dim=1)
            - 2 * torch.matmul(z_flattened, self.embedding.weight.t())
        )

        # Find nearest embedding index
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        quantized = self.embedding(encoding_indices).view(z.shape)

        # Compute losses
        e_latent_loss = F.mse_loss(quantized.detach(), z)
        q_latent_loss = F.mse_loss(quantized, z.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        # Straight-through estimator
        quantized = z + (quantized - z).detach()

        # Calculate codebook usage
        avg_probs = torch.histc(encoding_indices.float(), bins=self.num_embeddings) / encoding_indices.numel()

        return quantized, loss, avg_probs


# ------------------ 2. VQ-VAE Model ------------------
class VQVAE(nn.Module):
    def __init__(self, num_embeddings=128, embedding_dim=64, commitment_cost=0.25):
        super(VQVAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, embedding_dim, 3, 1, 1),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embedding_dim, 64, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 3, 1, 1),
            nn.Sigmoid(),
        )

        self.vq = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)

    def forward(self, x):
        z = self.encoder(x)
        quantized, vq_loss, codebook_usage = self.vq(z)
        x_recon = self.decoder(quantized)
        return x_recon, vq_loss, codebook_usage


# ------------------ 3. Training Loop ------------------
def main():
    # Config
    batch_size = 128
    epochs = 15
    lr = 1e-3
    num_embeddings = 128
    embedding_dim = 64
    commitment_cost = 0.25
    checkpoint_interval = 3
    device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs("outputs/vqvae_outputs", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    # Data
    train_loader, test_loader = get_dataloaders(batch_size=batch_size, augment=True)

    # Model, optimizer, logger
    model = VQVAE(num_embeddings, embedding_dim, commitment_cost).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    logger = TrainingLogger(folder="logs")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, total_recon, total_vq = 0, 0, 0

        for data, _ in train_loader:
            data = data.to(device)
            optimizer.zero_grad()

            recon, vq_loss, codebook_usage = model(data)
            recon_loss = F.mse_loss(recon, data)
            loss = recon_loss + vq_loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_vq += vq_loss.item()

        avg_loss = total_loss / len(train_loader)
        avg_recon = total_recon / len(train_loader)
        avg_vq = total_vq / len(train_loader)
        print(f"Epoch {epoch}/{epochs} | Total: {avg_loss:.4f} | Recon: {avg_recon:.4f} | VQ: {avg_vq:.4f}")

        logger.log(epoch, {"Total_Loss": avg_loss, "Recon_Loss": avg_recon, "VQ_Loss": avg_vq})

        if epoch % checkpoint_interval == 0 or epoch == epochs:
            save_checkpoint(model, optimizer, epoch, avg_loss, folder="checkpoints", prefix="vqvae")

        with torch.no_grad():
            data, _ = next(iter(test_loader))
            data = data.to(device)
            recon, _, _ = model(data)
            comparison = torch.cat([data[:8], recon[:8]])
            utils.save_image(comparison.cpu(), f"outputs/vqvae_outputs/reconstruction_epoch_{epoch}.png", nrow=8)

            print(f"Codebook usage avg (epoch {epoch}): {torch.sum(codebook_usage > 0).item()} / {num_embeddings}")

    # ------------------ 4. Save final results to CSV ------------------
    csv_path = "results/vqvae_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "Epochs", "Final_Total_Loss", "Final_Recon_Loss", "Final_VQ_Loss"])
        writer.writerow(["VQ-VAE", epochs, round(avg_loss, 4), round(avg_recon, 4), round(avg_vq, 4)])

    print(f"✅ Saved VQ-VAE results to {csv_path}")
    print("Training complete. Check 'outputs/vqvae_outputs/' for results and 'logs/' for metrics.")


if __name__ == "__main__":
    main()
