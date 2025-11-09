# train_gan.py
import torch
import torch.nn as nn
import torch.optim as optim
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
    epochs = 20
    lr = 0.0002
    latent_dim = 100
    device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs("outputs/gan_outputs", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    # ------------------ 2. Dataset ------------------
    train_loader, _ = get_dataloaders(batch_size=batch_size, augment=True)

    # ------------------ 3. Model Definitions ------------------
    class Generator(nn.Module):
        def __init__(self, latent_dim):
            super(Generator, self).__init__()
            self.model = nn.Sequential(
                nn.Linear(latent_dim, 256),
                nn.BatchNorm1d(256),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(256, 512),
                nn.BatchNorm1d(512),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(512, 1024),
                nn.BatchNorm1d(1024),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(1024, 28 * 28),
                nn.Tanh()  # output range [-1, 1]
            )

        def forward(self, z):
            img = self.model(z)
            return img.view(-1, 1, 28, 28)

    class Discriminator(nn.Module):
        def __init__(self):
            super(Discriminator, self).__init__()
            self.model = nn.Sequential(
                nn.Linear(28 * 28, 512),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(512, 256),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(256, 1),
                nn.Sigmoid()
            )

        def forward(self, img):
            img_flat = img.view(img.size(0), -1)
            return self.model(img_flat)

    # ------------------ 4. Initialize Models ------------------
    generator = Generator(latent_dim).to(device)
    discriminator = Discriminator().to(device)

    criterion = nn.BCELoss()
    optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    logger = TrainingLogger(folder="logs")

    # ------------------ 5. Training Loop ------------------
    for epoch in range(1, epochs + 1):
        generator.train()
        discriminator.train()
        total_g_loss, total_d_loss = 0, 0

        for real_imgs, _ in train_loader:
            real_imgs = real_imgs.to(device)
            real_imgs = (real_imgs - 0.5) / 0.5  # Normalize [0,1] -> [-1,1]

            # ---------------- Train Discriminator ----------------
            optimizer_D.zero_grad()

            real_labels = torch.ones(real_imgs.size(0), 1, device=device)
            real_output = discriminator(real_imgs)
            d_real_loss = criterion(real_output, real_labels)

            z = torch.randn(real_imgs.size(0), latent_dim, device=device)
            fake_imgs = generator(z)
            fake_labels = torch.zeros(real_imgs.size(0), 1, device=device)
            fake_output = discriminator(fake_imgs.detach())
            d_fake_loss = criterion(fake_output, fake_labels)

            d_loss = (d_real_loss + d_fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()

            # ---------------- Train Generator ----------------
            optimizer_G.zero_grad()
            z = torch.randn(real_imgs.size(0), latent_dim, device=device)
            generated_imgs = generator(z)
            output = discriminator(generated_imgs)
            g_loss = criterion(output, real_labels)
            g_loss.backward()
            optimizer_G.step()

            total_d_loss += d_loss.item()
            total_g_loss += g_loss.item()

        avg_d_loss = total_d_loss / len(train_loader)
        avg_g_loss = total_g_loss / len(train_loader)
        print(f"Epoch {epoch}/{epochs} | D Loss: {avg_d_loss:.4f} | G Loss: {avg_g_loss:.4f}")

        # Log losses
        logger.log(epoch, {"D_Loss": avg_d_loss, "G_Loss": avg_g_loss})

        # Save checkpoints every 2 epochs
        if epoch % 2 == 0 or epoch == epochs:
            save_checkpoint(generator, optimizer_G, epoch, avg_g_loss, folder="checkpoints", prefix="gan_gen")
            save_checkpoint(discriminator, optimizer_D, epoch, avg_d_loss, folder="checkpoints", prefix="gan_disc")

        # Save sample images
        with torch.no_grad():
            z = torch.randn(64, latent_dim, device=device)
            gen_imgs = generator(z).cpu()
            gen_imgs = (gen_imgs + 1) / 2  # Convert from [-1,1] to [0,1]
            save_image_batch(gen_imgs, filename=f"sample_epoch_{epoch}.png", folder="outputs/gan_outputs")

    # ------------------ 6. Save final results to CSV ------------------
    csv_path = "results/gan_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "Epochs", "Final_D_Loss", "Final_G_Loss"])
        writer.writerow(["GAN", epochs, round(avg_d_loss, 4), round(avg_g_loss, 4)])

    print(f"✅ Saved GAN results to {csv_path}")
    print("Training complete. Check 'outputs/gan_outputs/' for generated samples.")


if __name__ == "__main__":
    main()
