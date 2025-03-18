# __init__.py is called when imported by another Python file, where this directory is a python package, consisting modules of .py files

# Setting __all__ tells Python which modules (.py files) to import when importing this package
# __all__ = ['dne']

import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
import numpy as np
import torch


def show(imgs, titles=None, save_path=None):
    """
    Display one or more images with consistent styling regardless of IDE theme

    Args:
        imgs: A single tensor or list of tensors representing images
        titles: Optional list of titles for each image
        save_path: Optional path to save the figure
    """
    if not isinstance(imgs, list):
        imgs = [imgs]

    # ImageNet mean and std used by ViT transforms
    mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)

    # Create figure with white background
    plt.style.use('default')  # Reset to default style
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False, figsize=(4 * len(imgs), 4))
    fig.patch.set_facecolor('white')  # Set figure background to white

    for i, img in enumerate(imgs):
        img = img.detach().cpu()

        # Denormalize the image
        img = img * std + mean

        # Clamp values to valid range [0, 1]
        img = torch.clamp(img, 0, 1)

        # Convert to PIL Image
        img = F.to_pil_image(img)

        # Display image with white background
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        axs[0, i].set_facecolor('white')  # Set subplot background to white

        # Add title if provided
        if titles and i < len(titles):
            axs[0, i].set_title(titles[i], color='black')  # Ensure black text

    plt.tight_layout()

    if save_path:
        # Save with white background and high DPI
        plt.savefig(save_path, bbox_inches='tight', facecolor='white', dpi=300)

    plt.show()
    plt.close()  # Clean up
