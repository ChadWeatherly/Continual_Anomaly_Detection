"""
Class to create the Encoder network
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange  # Library for cleaner tensor reshaping operations
from ..ViT import ViT

class Encoder(ViT):
    """

    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, x, oasa_features):
        # Takes in image input of size (B, 3, 224, 224)
        # As well as the oasa_features produced from the discriminator
        # Returns either the features list or the final output

        out = super().forward(x, return_features=False, oasa_features=oasa_features)
        # out, and all tensors in features, have shape (B, L, E)
        # where
        # - L = sequence length = num_patches
        # - E = embedding_dimension = 64 (default)
        return out