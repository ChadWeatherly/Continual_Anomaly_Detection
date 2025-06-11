"""
Tools to create the discriminator network
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange  # Library for cleaner tensor reshaping operations
from ..ViT import ViT

class Discriminator(ViT):
    """

    """
    def __init__(self, output_size=12, **kwargs):
        super().__init__(return_feature_outputs=True, **kwargs)

        input_dim = (self.patch_dim ** 2) * self.embedding_dim
        self.mlp = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, input_dim*4),
            nn.GELU(),
            nn.Linear(input_dim*4, output_size),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        # Takes in image input of size (B, 3, 224, 224)

        # Since the discriminator forces return_feature_outputs=True,
        # it returns two items, the actual final layer output,
        # then a list of each layer's output tensor, detached
        out, features = super().forward(x)
        # out, and all tensors in features, have shape (B, L, E)
        # where
        # - L = sequence length = num_patches
        # - E = embedding_dimension = 64 (default)

        # Re-arrange for MLP
        out = rearrange(out, 'B L E -> B (L E)')
        out = self.mlp(out)

        return out, features
