"""
Base ViT class for IUF, used to create the
discriminator, encoder, and decoder.

Based on author's implementation (original_code/reconstructions/ViT.py),
and it should be noted that code experiments use different models for the encoder
(see original_code/experiments config.yaml files),
but the paper explicitly shows that ViT's are used for all 3 component models:
- Discriminator
- Encoder
- Decoder

ViT Block Diagram at bottom of page
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange  # Library for cleaner tensor reshaping operations

class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention mechanism that operates on 2D feature maps rather than
    tokenized sequences as in standard ViT implementations. Takes in an input
    of size (B x embed_dim x H x W), where the default embed_dim = 64.
    """
    def __init__(self, embed_dim, num_heads):
        super().__init__()

        # Ensure dimensions are compatible with multi-head mechanism
        assert embed_dim % num_heads == 0, "embedding dimension must be divisible by number of heads"

        self.num_heads = num_heads  # Number of attention heads
        self.head_dim = embed_dim // num_heads  # Dimension per head

        # Linear projections for query, key, value
        # Maintains input shape, but gives each pixel its own transformation
        # the Linear MLP takes a vector of size (*, H_in), where * means any number of dimensions
        # and outputs a vector of size (*, H_out)
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # x,q,k,v shape: (B x L x E)
        # Linear projections
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        # Reshape for each head
        q = rearrange(q, 'B L (n d) -> n B L d', n=self.num_heads)
        k = rearrange(k, 'B L (n d) -> n B L d', n=self.num_heads)
        v = rearrange(v, 'B L (n d) -> n B L d', n=self.num_heads)

        head_attn_scores = []
        for h in range(self.num_heads):
            # Vectors of size (B, L, embed_dim)
            # Represents L = HxW tokens, each of size embed_dim
            head_q = q[h]
            head_k = k[h]
            head_v = v[h]

            # returns matrix of size (L x L), where
            # each row in the matrix is the atten weight
            attn_weights = self.Softmax(torch.matmul(head_q, head_k.transpose(-2, -1)))



        return

class ViTBlock(nn.Module):
    """
    A single transformer encoder block with self-attention and MLP.
    """
    def __init__(self, embed_dim, num_heads):
        super().__init__()

        # Layer norm before attention
        self.norm1 = nn.LayerNorm(embed_dim)

        # Multi-head Self Attention and Final LayerNorm
        self.MHSA = MultiHeadSelfAttention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)  # Layer norm before MLP

    def forward(self, x):
        """
        Takes in a tensor of shape [B x L x embed_dim]
        and outputs a tensor of the same shape.
        """
        
        # First sub-block: Normalization + Self-Attention + Residual
        out = self.MHSA(self.norm1(x))  # Apply norm, then Multi-head Self Attention
        x = x + out                             # Residual connection

        # Second sub-block: Normalization + MLP + Residual
        x = x + self.mlp(self.norm2(x))         # Apply norm, MLP, then residual
        return x


# noinspection DuplicatedCode
class ViT(nn.Module):
    """
    Modified Vision Transformer that operates on each pixel, as opposed to patches

    All params are the same as original IUF paper, unless otherwise specified.
    Recall, the embed_dim is split across all heads, so embed_dim % num_heads must be 0.
    """
    def __init__(self, in_channels=3,
                 output_vec_dim=12,
                 patch_size=1,  # Not actually creating patches - operates directly on pixels
                 embed_dim=16,
                 num_heads=4,
                 num_layers=4):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = None
        self.embedding_dim = embed_dim

        # Initial convolutional embedding
        self.conv1 = nn.Conv2d(in_channels,
                               embed_dim,
                               kernel_size=patch_size,
                               stride=1)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.gelu = nn.GELU()

        # Stack of ViT blocks
        self.vit_blocks = nn.ModuleList([
            ViTBlock(embed_dim, num_heads)
            for _ in range(num_layers)  # 4 transformer blocks
        ])

        # Classification head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # Global average pooling
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, output_vec_dim)  # Final classification layer
        )

    def forward(self, x):
        # Expects images to be of shape (B, 3, 224, 224)

        # Get image embedding, where
        # Each pixel = a token
        out = self.conv1(x)  # (B, embed_dim, 224, 224), so each pixel has its own embedding
        out = self.ln1(out)
        out = self.gelu(out)
        # Rearrange for passing into ViT Blocks, leaving us with a sequence of tokens
        out = rearrange(out, 'B D H W -> B (H W) D') # (B, HxW = L, D)

        # Store intermediate outputs for potential visualization or anomaly detection
        outputs_map = []
        # Pass through transformer blocks
        for vit_block in self.vit_blocks:
            out = vit_block(out)  # [batch, height, width, embed_dim]
            outputs_map.append(out)  # Store intermediate representations

        # Rearrange back to [batch, channels, height, width] for CNN-style pooling
        out = rearrange(out, 'b h w c -> b c h w')

        # Global average pooling and classification
        out = self.avgpool(out)  # [batch, embed_dim, 1, 1]
        out = out.view(out.size(0), -1)  # [batch, embed_dim]
        out = self.fc(out)  # [batch, num_classes]

        # Detach intermediate outputs to prevent gradients from flowing through them
        # This is likely done to save memory when these are used for visualization
        outputs_map = [x.detach() for x in outputs_map]

        # Return dictionary with classification output and feature maps
        return {"class_out": out, "outputs_map": outputs_map}


"""
ViT Block Diagram based on author implemented method

B = Batch Size
C_in = Channels of input
H_in = Height of input
W_in = Width of input
D_emb = embed_dim = embedding dimension
H_map = Height of feature map
W_map = Width of feature map
N_heads = Number of heads in Multi-Head Attention
mlpp_dim = MLP dimension
N_blocks = Number of Transformer Blocks
N_classes = Number of classes

                               ┌────────────────────────────────────────────┐
                               │              Input Image (`x["image"]`)    │
                               │             (B × C_in × H_in × W_in)       │
                               └─────────────────────┬──────────────────────┘
                                                     │
                                                     ▼
                               ┌────────────────────────────────────────────┐
                               │      Initial Convolutional Embedding       │
                               │                                            │
                               │ 1. `conv1 = Conv2d(C_in, D_emb, ks=1)`     │
                               │    Output: (B × D_emb × H_map × W_map)     │
                               │ 2. `ln1 = LayerNorm2d(D_emb)`              │
                               │ 3. `relu = ReLU()`                         │
                               │    Output: (B × D_emb × H_map × W_map)     │
                               └─────────────────────┬──────────────────────┘
                                                     │
                                                     ▼
                               ┌────────────────────────────────────────────┐
                               │      Reshape for ViT Blocks (einops)       │
                               │ `rearrange(out, 'b c h w -> b h w c')`     │
                               │  Output: (B × H_map × W_map × D_emb)       │
                               └─────────────────────┬──────────────────────┘
                                                     │
                                                     ▼
                               ┌──────────────────────────────────────────────┐
                               │                 ViT Blocks                   │
                               │      (Process data in B H W D_emb format)    │
                               │                                              │
                               │    ┌──────────────────────────────────┐      │
                               │    │        ViT Block 1 of N_blocks   │      │
                               │    │  (Input: B × H_map × W_map × D_emb)│    │
                               │    │                                  │      │
                               │    │  ┌───────────────────────────┐   │      │
                               │    │  │  LayerNorm1 (on D_emb)    │   │      │
                               │    │  └────────────┬──────────────┘   │      │
                               │    │               │                  │      │
                               │    │               ▼                  │      │
                               │    │  ┌───────────────────────────┐   │      │
                               │    │  │ MultiHeadSelfAttention    │   │      │
                               │    │  │ (Custom: Spatial Attention) │ │      │
                               │    │  │  Input: B H_map W_map D_emb │ │      │
                               │    │  │  Output: B H_map W_map D_emb│ │      │
                               │    │  └────────────┬──────────────┘   │      │
                               │    │               │ Add              │      │
                               │    │               └─────►Residual 1  │      │
                               │    │                          │       │      │
                               │    │                          ▼       │      │
                               │    │  ┌───────────────────────────┐   │      │
                               │    │  │  LayerNorm2 (on D_emb)    │   │      │
                               │    │  └────────────┬──────────────┘   │      │
                               │    │               │                  │      │
                               │    │               ▼                  │      │
                               │    │  ┌───────────────────────────┐   │      │
                               │    │  │         MLP Block         │   │      │
                               │    │  │ Input: B H_map W_map D_emb│   │      │
                               │    │  │---------------------------│   │      │
                               │    │  │ Reshape: B H W D -> B (HW) D│ │      │
                               │    │  │ Linear(D_emb, mlp_dim)    │   │      │
                               │    │  │ GELU()                    │   │      │
                               │    │  │ Linear(mlp_dim, D_emb)    │   │      │
                               │    │  │ Reshape: B (HW) D -> B H W D│ │      │
                               │    │  │ (Note: Assumes H_map=W_map  │ │      │
                               │    │  │  for unflattening in code)│   │      │
                               │    │  │ Output: B H_map W_map D_emb│  │      │
                               │    │  └────────────┬──────────────┘   │      │
                               │    │               │ Add              │      │
                               │    │               └─────►Residual 2  │      │
                               │    │                          │       │      │
                               │    │                          ▼       │      │
                               │    │ Output: (B × H_map × W_map × D_emb)│    │
                               │    └──────────────────────────────────┘      │
                               │                   │                          │
                               │                   ▼                          │
                               │    ┌────────────────────────────────┐        │
                               │    │ More ViT Blocks (2 to N_blocks)│        │
                               │    │      (Same Structure)          │        │
                               │    └────────────────────────────────┘        │
                               │                   │                          │
                               │                   ▼                          │
                               │ Output after N_blocks: (B×H_map×W_map×D_emb) │
                               └─────────────────────┬────────────────────────┘
                                                     │
                                                     ▼
                               ┌────────────────────────────────────────────┐
                               │     Reshape for Pooling/Classification     │
                               │ `rearrange(out, 'b h w c -> b c h w')`     │
                               │  Output: (B × D_emb × H_map × W_map)       │
                               └─────────────────────┬──────────────────────┘
                                                     │
                                                     ▼
                               ┌────────────────────────────────────────────┐
                               │          Global Average Pooling            │
                               │  `AdaptiveAvgPool2d((1, 1))`               │
                               │  Output: (B × D_emb × 1 × 1)               │
                               └─────────────────────┬──────────────────────┘
                                                     │
                                                     ▼
                               ┌────────────────────────────────────────────┐
                               │           Flatten for Classifier           │
                               │      `out.view(out.size(0), -1)`           │
                               │        Output: (B × D_emb)                 │
                               └─────────────────────┬──────────────────────┘
                                                     │
                                                     ▼
                               ┌────────────────────────────────────────────┐
                               │             Classification Head (`fc`)     │
                               │                                            │
                               │ 1. `Linear(D_emb, D_emb)`                  │
                               │ 2. `ReLU()`                                │
                               │ 3. `Linear(D_emb, N_classes)`              │
                               │                                            │
                               │  Output (`class_out`): (B × N_classes)     │
                               └────────────────────────────────────────────┘

                               
       
       
Detailed Multi-Head Attention Block Diagram below:
https://docs.pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html

B = Batch Size
H_map = Height of the input feature map
W_map = Width of the input feature map
D_emb = dim = Embedding Dimension
N_h = num_heads = Number of attention heads
D_h = head_dim = D_emb / N_h = Dimension of each attention head
                       
                               ┌──────────────────────────────────┐
                               │ Input Feature Map (x)            │
                               │   [B, H_map, W_map, D_emb]       │
                               └─────────────────┬────────────────┘
                                                 │
                                                 ▼
                     ┌───────────────────────────────────────────────┐
                     │     Linear Projections (Query, Key, Value)    │
                     │                                               │
                     │ q = self.query(x)  -> [B, H_map, W_map, D_emb]│
                     │ k = self.key(x)    -> [B, H_map, W_map, D_emb]│
                     │ v = self.value(x)  -> [B, H_map, W_map, D_emb]│
                     └───────────┬───────────┬───────────┬───────────┘
                                 │ (q)       │ (k)       │ (v)
                                 ▼           ▼           ▼
              ┌───────────────────────────────────────────────────────────┐
              │ Reshape for Multi-Head Attention (using einops.rearrange) │
              │                                                           │
              │ q_h = rearrange(q, 'b hm wm (nh dh) -> b nh hm wm dh')     │
              │       Shape: [B, N_h, H_map, W_map, D_h]                  │
              │                                                           │
              │ k_h = rearrange(k, 'b hm wm (nh dh) -> b nh hm wm dh')     │
              │       Shape: [B, N_h, H_map, W_map, D_h]                  │
              │                                                           │
              │ v_h = rearrange(v, 'b hm wm (nh dh) -> b nh hm wm dh')     │
              │       Shape: [B, N_h, H_map, W_map, D_h]                  │
              └────────────────────┬───────────────────┬──────────────────┘
                                   │ (q_h)             │ (k_h)       │ (v_h)
                                   ▼                   ▼             |
              ┌─────────────────────────────────────────────────────┐ |
              │          Scaled Dot-Product Attention               │ |
              │ (Calculated for each head and each H_map position)  │ |
              │                                                     │ |
              │ 1. Transpose Key:                                   │ |
              │    k_h_T = k_h.transpose(-2, -1)                    │ |
              │            Shape: [B, N_h, H_map, D_h, W_map]       │ |
              │                                                     │ |
              │ 2. Matmul Q and K_T:                                │ |
              │    scores = matmul(q_h, k_h_T)                      │ |
              │             Shape: [B, N_h, H_map, W_map, W_map]    │ |
              │                                                     │ |
              │ 3. Scale Scores:                                    │ |
              │    scores = scores / sqrt(D_h)                      │ |
              │                                                     │ |
              │ 4. Softmax:                                         │ |
              │    attn_weights = softmax(scores, dim=-1)           │ |
              │                 Shape: [B, N_h, H_map, W_map, W_map]│ |
              └────────────────────┬────────────────────────────────┘ |
                                   │ (attn_weights)                   │ (v_h)
                                   ▼                                  ▼
              ┌───────────────────────────────────────────────────────────┐
              │             Apply Attention Weights to Values             │
              │                                                           │
              │ weighted_v = matmul(attn_weights, v_h)                    │
              │              Shape: [B, N_h, H_map, W_map, D_h]           │
              └────────────────────┬──────────────────────────────────────┘
                                   │
                                   ▼
              ┌───────────────────────────────────────────────────────────┐
              │      Reshape and Concatenate Heads (einops.rearrange)     │
              │                                                           │
              │ out_concat = rearrange(weighted_v, 'b nh hm wm dh -> b hm wm (nh dh)')│
              │              Shape: [B, H_map, W_map, D_emb]              │
              └────────────────────┬──────────────────────────────────────┘
                                   │
                                   ▼
                           ┌──────────────────────────────────┐
                           │      Final Linear Projection     │
                           │                                  │
                           │ output = self.out(out_concat)    │
                           │        Shape: [B, H_map, W_map, D_emb]│
                           └──────────────────────────────────┘


"""