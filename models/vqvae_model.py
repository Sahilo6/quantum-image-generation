import torch
import torch.nn as nn
import torch.nn.functional as F

# -------- Vector Quantizer -------- #
class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost

        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        self.embeddings.weight.data.uniform_(-1/self.num_embeddings, 1/self.num_embeddings)

    def forward(self, z):
        # Flatten input (B, C, H, W) -> (BHW, C)
        z_flattened = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z_flattened.view(-1, self.embedding_dim)

        # Compute distances between z and embeddings
        distances = (
            torch.sum(z_flattened ** 2, dim=1, keepdim=True)
            + torch.sum(self.embeddings.weight ** 2, dim=1)
            - 2 * torch.matmul(z_flattened, self.embeddings.weight.t())
        )

        # Find nearest embedding index for each vector
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.size(0), self.num_embeddings, device=z.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and reshape back
        quantized = torch.matmul(encodings, self.embeddings.weight)
        quantized = quantized.view(z.shape)

        # Straight-through estimator for gradients
        quantized = z + (quantized - z).detach()

        # Compute losses
        codebook_loss = F.mse_loss(quantized.detach(), z)
        commitment_loss = F.mse_loss(quantized, z.detach()) * self.commitment_cost
        loss = codebook_loss + commitment_loss

        return quantized, loss

# -------- Encoder-Decoder -------- #
class Encoder(nn.Module):
    def __init__(self, in_channels=1, hidden_channels=128, embedding_dim=64):
        super(Encoder, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 4, 2, 1),  # 28 -> 14
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, 4, 2, 1),  # 14 -> 7
            nn.ReLU(),
            nn.Conv2d(hidden_channels, embedding_dim, 3, 1, 1),
        )

    def forward(self, x):
        return self.net(x)

class Decoder(nn.Module):
    def __init__(self, embedding_dim=64, hidden_channels=128, out_channels=1):
        super(Decoder, self).__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(embedding_dim, hidden_channels, 4, 2, 1),  # 7 -> 14
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_channels, out_channels, 4, 2, 1),  # 14 -> 28
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)

# -------- Full VQ-VAE -------- #
class VQVAE(nn.Module):
    def __init__(self, num_embeddings=512, embedding_dim=64, commitment_cost=0.25):
        super(VQVAE, self).__init__()
        self.encoder = Encoder(1, 128, embedding_dim)
        self.quantizer = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
        self.decoder = Decoder(embedding_dim, 128, 1)

    def forward(self, x):
        z_e = self.encoder(x)
        z_q, vq_loss = self.quantizer(z_e)
        x_recon = self.decoder(z_q)
        return x_recon, vq_loss
