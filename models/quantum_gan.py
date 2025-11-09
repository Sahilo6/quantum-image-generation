"""
Quantum GAN Implementation
Two variants: Quantum Generator OR Quantum Discriminator
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.quantum_attention import QuantumAttentionLayer


class QuantumGenerator(nn.Module):
    """Generator with quantum layer: noise → quantum features → image"""
    
    def __init__(self, latent_dim=100, n_qubits=8, n_layers=3, 
                 entanglement='linear', output_dim=784):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.n_qubits = n_qubits
        
        # Classical preprocessing
        self.fc1 = nn.Linear(latent_dim, 128)
        self.fc2 = nn.Linear(128, n_qubits)
        
        # Quantum layer
        self.quantum_layer = QuantumAttentionLayer(
            n_qubits=n_qubits,
            n_layers=n_layers,
            entanglement=entanglement
        )
        
        # Classical postprocessing
        self.fc3 = nn.Linear(n_qubits, 256)
        self.fc4 = nn.Linear(256, 512)
        self.fc5 = nn.Linear(512, output_dim)
    
    def forward(self, z):
        # Preprocess noise
        h = F.relu(self.fc1(z))
        h = torch.tanh(self.fc2(h))
        
        # Quantum processing
        h = self.quantum_layer(h)
        
        # Generate image
        h = F.relu(self.fc3(h))
        h = F.relu(self.fc4(h))
        img = torch.sigmoid(self.fc5(h))
        
        return img.view(-1, 1, 28, 28)


class QuantumDiscriminator(nn.Module):
    """Discriminator with quantum layer: image features → quantum → classification"""
    
    def __init__(self, input_dim=784, n_qubits=8, n_layers=3, entanglement='linear'):
        super().__init__()
        
        self.n_qubits = n_qubits
        
        # Classical feature extraction
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, n_qubits)
        
        # Quantum layer
        self.quantum_layer = QuantumAttentionLayer(
            n_qubits=n_qubits,
            n_layers=n_layers,
            entanglement=entanglement
        )
        
        # Classification head
        self.fc4 = nn.Linear(n_qubits, 64)
        self.fc5 = nn.Linear(64, 1)
    
    def forward(self, x):
        # Flatten and extract features
        h = x.view(x.size(0), -1)
        h = F.leaky_relu(self.fc1(h), 0.2)
        h = F.leaky_relu(self.fc2(h), 0.2)
        h = torch.tanh(self.fc3(h))
        
        # Quantum processing
        h = self.quantum_layer(h)
        
        # Classification
        h = F.leaky_relu(self.fc4(h), 0.2)
        out = torch.sigmoid(self.fc5(h))
        
        return out


class ClassicalGenerator(nn.Module):
    """Classical generator for comparison"""
    
    def __init__(self, latent_dim=100, output_dim=784):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, output_dim)
    
    def forward(self, z):
        h = F.relu(self.fc1(z))
        h = F.relu(self.fc2(h))
        img = torch.sigmoid(self.fc3(h))
        return img.view(-1, 1, 28, 28)


class ClassicalDiscriminator(nn.Module):
    """Classical discriminator for comparison"""
    
    def __init__(self, input_dim=784):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)
    
    def forward(self, x):
        h = x.view(x.size(0), -1)
        h = F.leaky_relu(self.fc1(h), 0.2)
        h = F.leaky_relu(self.fc2(h), 0.2)
        out = torch.sigmoid(self.fc3(h))
        return out


class QuantumGAN:
    """
    Quantum GAN wrapper
    Supports: quantum generator, quantum discriminator, or both
    """
    
    def __init__(self, latent_dim=100, n_qubits=8, n_layers=3, 
                 entanglement='linear', quantum_mode='generator'):
        """
        Args:
            quantum_mode: 'generator', 'discriminator', or 'both'
        """
        self.latent_dim = latent_dim
        self.quantum_mode = quantum_mode
        
        # Create generator
        if quantum_mode in ['generator', 'both']:
            self.generator = QuantumGenerator(
                latent_dim, n_qubits, n_layers, entanglement
            )
        else:
            self.generator = ClassicalGenerator(latent_dim)
        
        # Create discriminator
        if quantum_mode in ['discriminator', 'both']:
            self.discriminator = QuantumDiscriminator(
                784, n_qubits, n_layers, entanglement
            )
        else:
            self.discriminator = ClassicalDiscriminator(784)
    
    def get_models(self):
        """Return generator and discriminator"""
        return self.generator, self.discriminator


# Test the Quantum GAN
if __name__ == "__main__":
    print("Testing Quantum GAN...")
    
    # Test Quantum Generator
    print("\n=== Quantum Generator ===")
    qgan = QuantumGAN(latent_dim=100, n_qubits=8, quantum_mode='generator')
    gen, disc = qgan.get_models()
    
    z = torch.randn(4, 100)
    fake_imgs = gen(z)
    print(f"Generated images shape: {fake_imgs.shape}")
    
    # Test discriminator
    pred = disc(fake_imgs)
    print(f"Discriminator output shape: {pred.shape}")
    
    # Test gradient flow
    loss = pred.sum()
    loss.backward()
    print(f"Generator has quantum layer: {hasattr(gen, 'quantum_layer')}")
    
    # Test Quantum Discriminator
    print("\n=== Quantum Discriminator ===")
    qgan2 = QuantumGAN(latent_dim=100, n_qubits=8, quantum_mode='discriminator')
    gen2, disc2 = qgan2.get_models()
    
    fake_imgs2 = gen2(z)
    pred2 = disc2(fake_imgs2)
    print(f"Discriminator output shape: {pred2.shape}")
    print(f"Discriminator has quantum layer: {hasattr(disc2, 'quantum_layer')}")
    
    print("\n✓ Quantum GAN working!")