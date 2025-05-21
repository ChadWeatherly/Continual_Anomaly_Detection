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
    tokenized sequences as in standard ViT implementations.
    """
    def __init__(self, dim, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.dim = dim  # Total embedding dimension
        self.num_heads = num_heads  # Number of attention heads
        self.head_dim = dim // num_heads  # Dimension per head

        # Ensure dimensions are compatible with multi-head mechanism
        assert dim % self.num_heads == 0, "embedding dimension must be divisible by number of heads"

        # Linear projections for query, key, value, and output
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)  # Final projection

    def forward(self, x):
        # x shape: [batch, height, width, embedding_dim]
        b, h, w, _ = x.size()

        # Linear projections
        q = self.query(x)  # [b, h, w, dim]
        k = self.key(x)    # [b, h, w, dim]
        v = self.value(x)  # [b, h, w, dim]

        # Reshape for multi-head attention - split embedding dim into num_heads x head_dim
        # NOTE: This differs from standard ViT which operates on token sequences
        q = rearrange(q, 'b h w (n d) -> b n h w d', n=self.num_heads)  # [b, num_heads, h, w, head_dim]
        k = rearrange(k, 'b h w (n d) -> b n h w d', n=self.num_heads)  # [b, num_heads, h, w, head_dim]
        v = rearrange(v, 'b h w (n d) -> b n h w d', n=self.num_heads)  # [b, num_heads, h, w, head_dim]

        # Transpose key for attention computation
        k_T = k.transpose(-2, -1)  # [b, num_heads, h, d, w]

        # Compute scaled dot-product attention
        scores = torch.matmul(q, k_T) / self.head_dim**0.5  # [b, num_heads, h, w, w]
        attention = F.softmax(scores, dim=-1)  # Softmax along last dimension (w)

        # Apply attention weights to values
        out = torch.matmul(attention, v)  # [b, num_heads, h, w, head_dim]

        # Reshape back to original format and combine heads
        out = rearrange(out, 'b n h w d -> b h w (n d)')  # [b, h, w, dim]

        # Final projection
        out = self.out(out)  # [b, h, w, dim]
        return out

class MLP(nn.Module):
    """
    Multi-Layer Perceptron used in transformer blocks.
    """
    def __init__(self, dim, hidden_dim):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),  # First linear layer
            nn.GELU(),                  # GELU activation as in original transformer
            nn.Linear(hidden_dim, dim),  # Second linear layer to restore dimensions
        )

    def forward(self, x):
        # Reshape to [batch, num_tokens, dim] for MLP
        # NOTE: Assumes height*width = num_tokens and reshapes accordingly
        x = rearrange(x, 'b h w d -> b (h w) d')  # [b, h*w, dim]
        x = self.mlp(x)                          # [b, h*w, dim]

        # POTENTIAL ISSUE: This assumes that h = w = sqrt(h*w), which may not always be true
        # Reshape back to [batch, height, width, dim]
        x = rearrange(x, 'b (h w) d -> b h w d', h=int(x.size(1) ** 0.5))
        return x

class ViTBlock(nn.Module):
    """
    A single transformer encoder block with self-attention and MLP.
    """
    def __init__(self, dim, num_patches, hidden_dim, num_heads, mlp_dim):
        super(ViTBlock, self).__init__()
        self.norm1 = nn.LayerNorm(dim)  # Layer norm before attention
        self.self_attention = MultiHeadSelfAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)  # Layer norm before MLP
        self.mlp = MLP(dim, mlp_dim)    # MLP block

    def forward(self, x):
        # First sub-block: Normalization + Self-Attention + Residual
        out = self.self_attention(self.norm1(x))  # Apply norm, then attention
        x = x + out                             # Residual connection

        # Second sub-block: Normalization + MLP + Residual
        x = x + self.mlp(self.norm2(x))         # Apply norm, MLP, then residual
        return x

class ViT(nn.Module):
    """
    Modified Vision Transformer that operates on feature maps rather than
    using the standard patch-embedding approach.
    """
    def __init__(self, inplanes=3, num_classes=12, hidden_dim=256, num_heads=4, mlp_dim=128):
        super(ViT, self).__init__()
        self.patch_size = 1  # Not actually creating patches - operates directly on pixels
        self.num_patches = inplanes * 14 * 14  # POSSIBLE ISSUE: This calculation seems arbitrary
        self.embedding_dim = hidden_dim

        # Initial convolutional embedding - more like a CNN than standard ViT
        self.conv1 = nn.Conv2d(inplanes, hidden_dim, kernel_size=self.patch_size,
                               stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_dim)  # BatchNorm is unusual in ViT (usually uses LayerNorm)
        self.relu = nn.ReLU(inplace=True)      # ReLU is also different from standard ViT

        # Stack of ViT blocks
        self.vit_blocks = nn.ModuleList([
            ViTBlock(hidden_dim, self.num_patches, hidden_dim, num_heads, mlp_dim)
            for _ in range(4)  # 4 transformer blocks
        ])

        # Classification head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # Global average pooling
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_classes)  # Final classification layer
        )

    def forward(self, x):
        # Extract image from input dictionary
        out = self.conv1(x["image"])  # [batch, hidden_dim, h, w]
        out = self.bn1(out)
        out = self.relu(out)

        # Store intermediate outputs for potential visualization or anomaly detection
        outputs_map = []

        # Rearrange from [batch, channels, height, width] to [batch, height, width, channels]
        # for compatibility with ViT blocks that expect this format
        out = rearrange(out, 'b c h w -> b h w c')

        # Pass through transformer blocks
        for vit_block in self.vit_blocks:
            out = vit_block(out)  # [batch, height, width, hidden_dim]
            outputs_map.append(out)  # Store intermediate representations

        # Rearrange back to [batch, channels, height, width] for CNN-style pooling
        out = rearrange(out, 'b h w c -> b c h w')

        # Global average pooling and classification
        out = self.avgpool(out)  # [batch, hidden_dim, 1, 1]
        out = out.view(out.size(0), -1)  # [batch, hidden_dim]
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
D_emb = hidden_dim = embedding dimension
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
                               │ 2. `bn1 = BatchNorm2d(D_emb)`              │
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