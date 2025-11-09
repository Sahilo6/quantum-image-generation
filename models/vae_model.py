import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    """
    Variational Autoencoder for MNIST (28x28 images).
    """

    def __init__(self, latent_dim=20):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim

        # Encoder: Converts image -> latent mean & logvar
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1),   # (B, 1, 28, 28) -> (B, 32, 14, 14)
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),  # (B, 32, 14, 14) -> (B, 64, 7, 7)
            nn.ReLU(),
            nn.Flatten()
        )

        self.fc_mu = nn.Linear(64 * 7 * 7, latent_dim)
        self.fc_logvar = nn.Linear(64 * 7 * 7, latent_dim)

        # Decoder: Converts latent vector -> image
        self.fc_decode = nn.Linear(latent_dim, 64 * 7 * 7)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1),  # (B, 64, 7, 7) -> (B, 32, 14, 14)
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, 2, 1),   # (B, 32, 14, 14) -> (B, 1, 28, 28)
            nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick: z = mu + std * eps
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # Encode
        x = self.encoder(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)

        # Reparameterize
        z = self.reparameterize(mu, logvar)

        # Decode
        x = self.fc_decode(z)
        x = x.view(-1, 64, 7, 7)
        recon = self.decoder(x)

        return recon, mu, logvar


def vae_loss(recon_x, x, mu, logvar, beta=1.0):
    """
    VAE Loss = Reconstruction + Beta * KL Divergence
    """
    recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return (recon_loss + beta * kl_div) / x.size(0)
