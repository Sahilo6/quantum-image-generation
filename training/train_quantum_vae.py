# train_quantum_vae.py
import torch
import torch.optim as optim
from torchvision import utils
from data_loader import get_dataloaders
from models.quantum_vae import QuantumAttentionVAE
from logger import TrainingLogger
from checkpoint import save_checkpoint
import os

# ---------------- Configuration ----------------
device = "cuda" if torch.cuda.is_available() else "cpu"
epochs = 5
batch_size = 16
lr = 1e-3
beta = 1.0
n_qubits = 8
n_layers = 3

# ---------------- Data ----------------
train_loader, test_loader = get_dataloaders(batch_size=batch_size, augment=False)

# ---------------- Model ----------------
model = QuantumAttentionVAE(
    input_dim=784,
    latent_dim=20,
    n_qubits=n_qubits,
    n_layers=n_layers,
    use_quantum=True
).to(device)

optimizer = optim.Adam(model.parameters(), lr=lr)
logger = TrainingLogger(folder="logs")

os.makedirs("outputs/quantum_vae_outputs", exist_ok=True)

# ---------------- Training Loop ----------------
for epoch in range(1, epochs + 1):
    model.train()
    total_loss, recon_loss_sum, kl_loss_sum = 0, 0, 0

    for data, _ in train_loader:
        data = data.to(device)
        optimizer.zero_grad()

        x_recon, mu, logvar = model(data)
        loss, recon_loss, kl_loss = model.loss_function(x_recon, data, mu, logvar, beta)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        recon_loss_sum += recon_loss.item()
        kl_loss_sum += kl_loss.item()

    avg_total = total_loss / len(train_loader)
    avg_recon = recon_loss_sum / len(train_loader)
    avg_kl = kl_loss_sum / len(train_loader)

    print(f"Epoch {epoch}/{epochs} | Total: {avg_total:.4f} | Recon: {avg_recon:.4f} | KL: {avg_kl:.4f}")
    logger.log(epoch, {"Total": avg_total, "Recon": avg_recon, "KL": avg_kl})

    # Save samples
    with torch.no_grad():
        samples = model.sample(num_samples=16, device=device)
        utils.save_image(samples, f"outputs/quantum_vae_outputs/samples_epoch_{epoch}.png", nrow=4)

    # Checkpoint every 2 epochs
    if epoch % 2 == 0:
        save_checkpoint(model, optimizer, epoch, avg_total, folder="checkpoints", prefix="quantum_vae")


print("\nTraining complete! Check 'outputs/quantum_vae_outputs/' for samples and 'logs/' for loss logs.")
