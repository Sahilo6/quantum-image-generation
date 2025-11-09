"""
Quantum Attention VAE Implementation
Classical Encoder → Quantum Attention → Classical Decoder
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.quantum_attention import QuantumAttentionLayer


class Encoder(nn.Module):
    """Classical encoder for VAE"""
    
    def __init__(self, input_dim=784, hidden_dim=256, latent_dim=20):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
    
    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class Decoder(nn.Module):
    """Classical decoder for VAE"""
    
    def __init__(self, latent_dim=20, hidden_dim=256, output_dim=784):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, z):
        h = F.relu(self.fc1(z))
        h = F.relu(self.fc2(h))
        x_recon = torch.sigmoid(self.fc3(h))
        return x_recon


class QuantumAttentionVAE(nn.Module):
    """
    Quantum Attention VAE
    Architecture: Encoder → Quantum Attention → Decoder
    """
    
    def __init__(self, input_dim=784, hidden_dim=256, latent_dim=20, 
                 n_qubits=8, n_layers=3, entanglement='linear', use_quantum=True):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.use_quantum = use_quantum
        self.n_qubits = n_qubits
        
        # Classical encoder
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        
        # Quantum attention layer (operates on latent space)
        if use_quantum:
            # Project latent to quantum dimension
            self.latent_to_quantum = nn.Linear(latent_dim, n_qubits)
            self.quantum_attention = QuantumAttentionLayer(
                n_qubits=n_qubits, 
                n_layers=n_layers, 
                entanglement=entanglement
            )
            self.quantum_to_latent = nn.Linear(n_qubits, latent_dim)
        
        # Classical decoder
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim)
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        # Flatten input
        batch_size = x.shape[0]
        x_flat = x.view(batch_size, -1)
        
        # Encode
        mu, logvar = self.encoder(x_flat)
        z = self.reparameterize(mu, logvar)
        
        # Quantum attention on latent space
        if self.use_quantum:
            z_quantum = self.latent_to_quantum(z)
            z_quantum = self.quantum_attention(z_quantum)
            z = self.quantum_to_latent(z_quantum)
        
        # Decode
        x_recon = self.decoder(z)
        x_recon = x_recon.view(batch_size, 1, 28, 28)
        
        return x_recon, mu, logvar
    
    def loss_function(self, x_recon, x, mu, logvar, beta=1.0):
        """VAE loss: reconstruction + KL divergence"""
        # Normalize input to [0, 1] if needed
        x_normalized = (x - x.min()) / (x.max() - x.min() + 1e-8)
        
        # Reconstruction loss (BCE for normalized data)
        recon_loss = F.binary_cross_entropy(
            x_recon.view(-1, 784), 
            x_normalized.view(-1, 784), 
            reduction='sum'
        )
        
        # KL divergence
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        return recon_loss + beta * kl_loss, recon_loss, kl_loss
    
    def sample(self, num_samples, device='cpu'):
        """Generate samples from the model"""
        with torch.no_grad():
            z = torch.randn(num_samples, self.latent_dim).to(device)
            
            if self.use_quantum:
                z_quantum = self.latent_to_quantum(z)
                z_quantum = self.quantum_attention(z_quantum)
                z = self.quantum_to_latent(z_quantum)
            
            samples = self.decoder(z)
            samples = samples.view(num_samples, 1, 28, 28)
        
        return samples


# Test the Quantum VAE
if __name__ == "__main__":
    print("Testing Quantum Attention VAE...")
    
    # Create model
    model = QuantumAttentionVAE(
        input_dim=784,
        latent_dim=20,
        n_qubits=8,
        n_layers=3,
        use_quantum=True
    )
    
    # Test forward pass with normalized data
    x = torch.rand(4, 1, 28, 28)  # Use rand instead of randn for [0,1] range
    x_recon, mu, logvar = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Input range: [{x.min():.3f}, {x.max():.3f}]")
    print(f"Reconstruction shape: {x_recon.shape}")
    print(f"Reconstruction range: [{x_recon.min():.3f}, {x_recon.max():.3f}]")
    print(f"Latent mu shape: {mu.shape}")
    print(f"Latent logvar shape: {logvar.shape}")
    
    # Test loss
    loss, recon_loss, kl_loss = model.loss_function(x_recon, x, mu, logvar)
    print(f"\nTotal loss: {loss.item():.4f}")
    print(f"Reconstruction loss: {recon_loss.item():.4f}")
    print(f"KL loss: {kl_loss.item():.4f}")
    
    # Test gradient flow
    loss.backward()
    if model.use_quantum:
        print(f"Quantum gradients: {model.quantum_attention.weights.grad is not None}")
    
    # Test sampling
    samples = model.sample(num_samples=8)
    print(f"\nGenerated samples shape: {samples.shape}")
    
    print("\n✓ Quantum VAE working!")