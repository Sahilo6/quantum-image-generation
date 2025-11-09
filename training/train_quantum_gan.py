# train_quantum_gan.py
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import utils
from data_loader import get_dataloaders
from models.quantum_gan import QuantumGAN
from logger import TrainingLogger
from checkpoint import save_checkpoint
import os

# ---------------- Configuration ----------------
device = "cuda" if torch.cuda.is_available() else "cpu"
epochs = 5
batch_size = 32
lr = 2e-4
latent_dim = 100
n_qubits = 8
n_layers = 3
quantum_mode = "generator"  # or "discriminator" or "both"

# ---------------- Data ----------------
train_loader, _ = get_dataloaders(batch_size=batch_size, augment=False)

# ---------------- Model Setup ----------------
qgan = QuantumGAN(latent_dim=latent_dim, n_qubits=n_qubits, n_layers=n_layers, quantum_mode=quantum_mode)
generator, discriminator = qgan.get_models()

generator.to(device)
discriminator.to(device)

optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
criterion = nn.BCELoss()
logger = TrainingLogger(folder="logs")

os.makedirs("outputs/quantum_gan_outputs", exist_ok=True)

# ---------------- Training Loop ----------------
for epoch in range(1, epochs + 1):
    generator.train()
    discriminator.train()
    total_d_loss, total_g_loss = 0, 0

    for real_imgs, _ in train_loader:
        real_imgs = real_imgs.to(device)
        batch_size = real_imgs.size(0)

        valid = torch.ones((batch_size, 1), device=device)
        fake = torch.zeros((batch_size, 1), device=device)

        # -------- Train Generator --------
        optimizer_G.zero_grad()
        z = torch.randn(batch_size, latent_dim, device=device)
        gen_imgs = generator(z)
        g_loss = criterion(discriminator(gen_imgs), valid)
        g_loss.backward()
        optimizer_G.step()

        # -------- Train Discriminator --------
        optimizer_D.zero_grad()
        real_loss = criterion(discriminator(real_imgs), valid)
        fake_loss = criterion(discriminator(gen_imgs.detach()), fake)
        d_loss = 0.5 * (real_loss + fake_loss)
        d_loss.backward()
        optimizer_D.step()

        total_d_loss += d_loss.item()
        total_g_loss += g_loss.item()

    avg_d_loss = total_d_loss / len(train_loader)
    avg_g_loss = total_g_loss / len(train_loader)

    print(f"Epoch {epoch}/{epochs} | D Loss: {avg_d_loss:.4f} | G Loss: {avg_g_loss:.4f}")
    logger.log(epoch, {"D_Loss": avg_d_loss, "G_Loss": avg_g_loss})

    # Save sample images
    with torch.no_grad():
        z = torch.randn(16, latent_dim, device=device)
        samples = generator(z)
        utils.save_image(samples, f"outputs/quantum_gan_outputs/samples_epoch_{epoch}.png", nrow=4, normalize=True)

    # Checkpoint every 2 epochs
    if epoch % 2 == 0:
        save_checkpoint(generator, optimizer_G, epoch, avg_g_loss, folder="checkpoints", prefix=f"quantum_gan_G")
        save_checkpoint(discriminator, optimizer_D, epoch, avg_d_loss, folder="checkpoints", prefix=f"quantum_gan_D")

print("\nTraining complete! Check 'outputs/quantum_gan_outputs/' for generated images.")
