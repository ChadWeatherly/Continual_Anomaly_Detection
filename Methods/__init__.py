# __init__.py is called when imported by another Python file, where this directory is a python package, consisting modules of .py files

# Setting __all__ tells Python which modules (.py files) to import when importing this package
# __all__ = ['dne']

import numpy as np
import torch
import torch.nn as nn
from torchvision.models.vision_transformer import vit_b_16
from torchvision.models import ViT_B_16_Weights
from torch.distributions.normal import Normal
import torch.nn.functional as F
from .DNE.dne import DNE_Model

class BaseAnomalyDetector(nn.Module):
    """Base class for anomaly detection models in continual learning scenarios.

    This class provides common functionality for:
    - Setting up device management
    - Base model construction
    - Training/inference patterns
    - Memory management for continual learning
    """
    def __init__(self, device=None):
        """Initialize the base anomaly detector.

        Args:
            backbone (str): Name of the backbone model to use
            pretrained (bool): Whether to use pretrained weights
            device (str): Device to use (defaults to best available)
        """
        super().__init__()

        # Set up device
        self.device = device or (
            "cuda" if torch.cuda.is_available() else
            "mps" if torch.backends.mps.is_available() else
            "cpu"
        )
        print(f"Using {self.device} device")

        # Move to device
        self.to(self.device)

    def forward(self, x):
        """Forward pass through the model.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Output tensor
        """
        raise NotImplementedError("Subclasses must implement forward method")

    def embed(self, x):
        """Extract embeddings from the input tensor.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Embeddings tensor
        """
        raise NotImplementedError("Subclasses must implement embed method")

    def train_one_epoch(self, dataloader, optimizer, criterion, task_num, **kwargs):
        """Train the model for one epoch.

        Args:
            dataloader (torch.utils.data.DataLoader): Dataloader for training data
            optimizer (torch.optim.Optimizer): Optimizer
            criterion (callable): Loss function
            task_num (int): Current task number
            **kwargs: Additional keyword arguments

        Returns:
            float: Average loss for the epoch
        """
        raise NotImplementedError("Subclasses must implement train_one_epoch method")

    def predict(self, x):
        """Make prediction for input.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Prediction
        """
        raise NotImplementedError("Subclasses must implement predict method")

    def save(self, path):
        """Save model state.

        Args:
            path (str): Path to save model state
        """
        torch.save({
            'model_state_dict': self.state_dict(),
            'memory': self.memory
        }, path)

    def load(self, path):
        """Load model state.

        Args:
            path (str): Path to load model state from
        """
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['model_state_dict'])



