# __init__.py is called when imported by another Python file, where this directory is a python package, consisting modules of .py files

# Setting __all__ tells Python which modules (.py files) to import when importing this package
# __all__ = ['dne']

import numpy as np
import os
import pandas as pd
import torch
import torch.nn as nn
from torchvision.models.vision_transformer import vit_b_16
from torchvision.models import ViT_B_16_Weights
from torch.distributions.normal import Normal
import torch.nn.functional as F

class BaseAnomalyDetector(nn.Module):
    """Base class for anomaly detection models in continual learning scenarios.

    This class provides common functionality for:
    - Setting up device management
    - Base model construction
    - Training/inference patterns
    - Memory management for continual learning
    """
    def __init__(self):
        """
        Initialize the base anomaly detector.
        """
        super().__init__()

        # Set up device
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

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
            float: Total Loss for that epoch
        """
        raise NotImplementedError("Subclasses must implement train_one_epoch method")

    def eval_one_epoch(self, dataloader, criterion, task_num, **kwargs):
        """
        Evaluate the model for one epoch of a testing set

        Args:
            dataloader:
            criterion:
            task_num:
            **kwargs:

        Returns:
            float: Total Loss for that epoch
        """
        raise NotImplementedError("Subclasses must implement eval_one_epoch method")

    def calc_results(self, dataset, exp, metrics):
        """
        Method used to calculate results for a given model and save that data as CSV
        Args:
            dataset (str): 'MTD' or 'MVTEC'
            exp (str): 'unsupervised' or 'supervised'
        Returns:

        """

        raise NotImplementedError("Subclasses must implement calc_results method")

    def predict(self, x):
        """Make prediction for input.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Prediction
        """
        raise NotImplementedError("Subclasses must implement predict method")

    def save(self, path):
        """
        Saves the model to disk.
        Args:
            path: path to save the model to.
        """
        torch.save(self.state_dict(), path)
        return

    def load(self, path):
        """
        Loads the model from disk.
        Args:
            path: path to load the model from.
        """
        self.load_state_dict(torch.load(path))
        return



