
# utils.py
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import os

# -------------------------------------------------------
# Function 1: Display a batch of images
# -------------------------------------------------------
def show_image_batch(images, title="Image Batch", nrow=8):
    """
    Displays a grid of images from a batch tensor.
    Args:
        images (Tensor): shape (B, C, H, W)
        title (str): title for the plot
        nrow (int): number of images per row
    """
    plt.figure(figsize=(10, 5))
    grid = vutils.make_grid(images[:nrow * (len(images)//nrow)], nrow=nrow, normalize=True)
    plt.imshow(grid.permute(1, 2, 0))
    plt.title(title)
    plt.axis('off')
    plt.show()


# -------------------------------------------------------
# Function 2: Save a batch of images to a file
# -------------------------------------------------------
def save_image_batch(images, filename="batch.png", nrow=8, folder="saved_batches"):
    """
    Saves a grid of images as a single PNG file.
    Args:
        images (Tensor): shape (B, C, H, W)
        filename (str): output file name
        nrow (int): number of images per row
        folder (str): folder to save images in
    """
    os.makedirs(folder, exist_ok=True)
    grid = vutils.make_grid(images, nrow=nrow, normalize=True)
    save_path = os.path.join(folder, filename)
    plt.imsave(save_path, grid.permute(1, 2, 0).cpu().numpy())
    print(f"✅ Saved image batch to: {save_path}")


# -------------------------------------------------------
# Optional test (run this file directly)
# -------------------------------------------------------
if __name__ == "__main__":
    import torch
    from data_loader import get_dataloaders

    train_loader, _ = get_dataloaders(batch_size=16, augment=True)
    images, labels = next(iter(train_loader))

    # Show and save a batch
    show_image_batch(images, title="Augmented MNIST Batch")
    save_image_batch(images, filename="example_batch.png")
