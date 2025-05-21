"""
Base ViT class for IUF, used to create the
discriminator, encoder, and decoder.

Based on author's implementation, and it should be noted that code experiments
use different models for the encoder (see original_code/experiments config.yaml files),
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
ViT Block Diagram based on forward() method in:
https://github.com/pytorch/vision/blob/main/torchvision/models/vision_transformer.py#L86

B = Batch Size
C = Channels
H = Height
W = Width
P = Patch Size
P×P = P^2 = S =Sequence Length
hidden_dim = embed_dim = 768
CLS Token = Class Token

                               ┌────────────────────────────────────────────┐
                               │              Input Image                   │
                               │                 (B×C×H×W)                  │
                               └─────────────────────┬──────────────────────┘
                                                     │
                               _process_input()      ▼
                               ┌────────────────────────────────────────────┐
                               │          Patch Embedding (conv_proj)       │
                               │                   Conv2d                   │
                               |      Outputs patch sequence of size:       |      
                               │          (B × (P×P) × hidden_dim),         │
                               └─────────────────────┬──────────────────────┘
                                                     │
                                                     ▼
                              Adding CLS token to sequence embedding
┌────────────────────────┐    ┌────────────────────────────────────────────┐
│   Class Token (CLS)    │───►│ Concatenate CLS Token with Patch Embeddings│
│     (B×1×embed_dim)    │    │           (B×(S+1)×embed_dim)              │
└────────────────────────┘    └─────────────────────┬──────────────────────┘
                                                    │

                                                    │
                                                    ▼
                               ┌────────────────────────────────────────────┐
                               │                 Encoder                    │
┌────────────────────────┐     │                                            │
│  Position Embeddings   │-───►│    ┌────────────────────────────────┐      │              
│  (1×(S+1)×embed_dim)   │     │    │    Add Position Embeddings     │      │
└────────────────────────┘     │    │       (B×(S+1)×embed_dim)      │      │
                               │    └──────────────┬────────────────-┘      │
                               │                   │                        │
                               │                   ▼                        │
                               │    ┌────────────────────────────────┐      │              
                               │    │         Add Dropout            │      │
                               │    └──────────────┬────────────────-┘      │
                               │                   │                        │
                               │                   ▼                        │
                               │    ┌────────────────────────────────┐      │
                               │    │   Encoder/Transformer Block 1  │      │
                               │    │  ┌─────────────────────────┐   │      │
                               │    │  │      Layer Norm 1       │   │      │
                               │    │  └───────────┬─────────────┘   │      │
                               │    │              │                 │      │
                               │    │              ▼                 │      │
                               │    │  ┌─────────────────────────┐   │      │
                               │    │  │   Multi-Head Attention  │   │      │
                               │    │  └───────────┬─────────────┘   │      │
                               │    │              │                 │      │
                               │    │              ▼                 │      │
                               │    │  ┌─────────────────────────┐   │      │
                               │    │  │   Residual Connection   │   │      │
                               │    │  └───────────┬─────────────┘   │      │
                               │    │              │                 │      │
                               │    │              ▼                 │      │
                               │    │  ┌─────────────────────────┐   │      │
                               │    │  │      Layer Norm 2       │   │      │
                               │    │  └───────────┬─────────────┘   │      │
                               │    │              │                 │      │
                               │    │              ▼                 │      │
                               │    │  ┌─────────────────────────┐   │      │
                               │    │  │         MLP Block       │   │      │
                               │    │  │   (Linear→GELU→Linear)  │   │      │
                               │    │  └───────────┬─────────────┘   │      │
                               │    │              │                 │      │
                               │    │              ▼                 │      │
                               │    │  ┌─────────────────────────┐   │      │
                               │    │  │   Residual Connection   │   │      │
                               │    │  └───────────┬─────────────┘   │      │
                               │    └──────────────┬────────────────-┘      │
                               │                   │                        │
                               │                   ▼                        │
                               │    ┌────────────────────────────── ┐       │
                               │    │             More              │       │
                               │    │ Encoder/Transformer Blocks ...│       │
                               │    │                               │       │
                               │    └──────────────┬────────────────┘       │
                               │                   │                        │
                               │                   ▼                        │
                               │    ┌────────────────────────────────┐      │
                               │    │ Encoder/Transformer Block N    │      │
                               │    │         (Same Structure)       │      │
                               │    └──────────────┬──────────────── ┘      │
                               │                   │                        │
                               │              layer_norm()                  │
                               │                                            │
                               │       Output: Same size as input           │
                               │           (B×(S+1)×embed_dim)              │
                               └───────────────────┬────────────────────────┘
                                                   │
                                                   ▼
                               ┌────────────────────────────────────────────┐
                               │        Extract CLS Token Embedding         │
                               │     from position 0 of the sequence        │
                               │               (B×embed_dim)                │
                               └─────────────────────┬──────────────────────┘
                                                     │
                                                     ▼
                               ┌────────────────────────────────────────────┐
                               │             Classification Head            │
                               │         Linear→Tanh→Linear → Softmax       │
                               │              (B×num_classes)               │
                               └────────────────────────────────────────────┘
                               
       
       
Detailed Multi-Head Attention Block Diagram below:
https://docs.pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html

S = Sequence Length
B = Batch Size
E = Embedding Dimension
H = Number of attention heads
D_h = Dimension of each head = E / H
                       
                                        ┌───────────────────────────────┐
                                        │     INPUTS TO FORWARD()       │
                                        │                               │
┌───────────────────┐ ┌───────────────┐ │ ┌───────────────┐             │
│ Query Tensor (Q)  │ │ Key Tensor (K)│ │ │Value Tensor(V)│             │
│ (S, B, E)         │ │ (S, B, E)     │ │ │ (S, B, E)     │             │
└─────────┬─────────┘ └───────┬───────┘ │ └───────┬───────┘             │
          │                   │         │         │                     │
          │                   │         │         │  ┌─────────────┐    │
          │                   │         │         │  │ key_padding │    │
          │                   │         │         │  │    mask     │    │
          │                   │         │         │  │ (Optional)  │    │
          │                   │         │         │  └─────┬───────┘    │
          │                   │         │         │        │            │
          │                   │         │         │  ┌─────▼───────┐    │
          │                   │         │         │  │    attn     │    │
          │                   │         │         │  │    mask     │    │
          │                   │         │         │  │ (Optional)  │    │
          │                   │         │         │  └─────────────┘    │
          │                   │         │         │                     │
          │                   │         └─────────┼─────────────────────┘
          │                   │                   │
          ▼                   ▼                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│                    MULTI-HEAD ATTENTION MECHANISM                       │
│                                                                         │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │                     LINEAR PROJECTIONS                            │  │
│  │                                                                   │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐             │  │
│  │  │  Q_proj(Q)   │  │  K_proj(K)   │  │  V_proj(V)   │             │  │
│  │  │  Linear Layer│  │  Linear Layer│  │  Linear Layer│             │  │
│  │  │  E → E       │  │  E → E       │  │  E → E       │             │  │
│  │  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘             │  │
│  │         │                 │                 │                     │  │
│  │         ▼                 ▼                 ▼                     │  │
│  │  ┌──────────────────────────────────────────────────────────┐     │  │
│  │  │             RESHAPE INTO MULTIPLE HEADS                  │     │  │
│  │  │                                                          │     │  │
│  │  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │     │  │
│  │  │  │  Q_heads     │  │  K_heads     │  │  V_heads     │    │     │  │
│  │  │  │              │  │              │  │              │    │     │  │
│  │  │  │(B, H, S, D_h)│  │(B, H, S, D_h)│  │(B, H, S, D_h)│    │     │  │
│  │  │  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘    │     │  │
│  │  └─────────┼───────────────┬─┼───────────────┬─┘────────────┘     │  │
│  └────────────┼───────────────┘ └───────────────┘────────────────────┘  │
│               │                                 │                       │
│               │                                 │                       │
│  ┌────────────▼─────────────────────────────────▼────────────────────┐  │
│  │                   ATTENTION CALCULATION                           │  │
│  │                                                                   │  │
│  │  ┌──────────────────────────────────────────────────────┐         │  │
│  │  │              SCALED DOT-PRODUCT                      │         │  │
│  │  │                                                      │         │  │
│  │  │  attn_weights = matmul(Q_heads, K_heads.transpose)   │         │  │
│  │  │                / sqrt(D_h)                           │         │  │
│  │  │                                                      │         │  │
│  │  │  (B, H, L, S)                                        │         │  │
│  │  └──────────────────────────┬───────────────────────────┘         │  │
│  │                             │                                     │  │
│  │                             ▼                                     │  │
│  │  ┌──────────────────────────────────────────────────────┐         │  │
│  │  │              APPLY MASKS (IF ANY)                    │         │  │
│  │  │                                                      │         │  │
│  │  │  - Apply key_padding_mask to ignore padded positions │         │  │
│  │  │  - Apply attn_mask for causal attention if needed    │         │  │
│  │  │                                                      │         │  │
│  │  └──────────────────────────┬───────────────────────────┘         │  │
│  │                             │                                     │  │
│  │                             ▼                                     │  │
│  │  ┌──────────────────────────────────────────────────────┐         │  │
│  │  │                   SOFTMAX                            │         │  │
│  │  │                                                      │         │  │
│  │  │  attn_weights = softmax(attn_weights, dim=-1)        │         │  │
│  │  │                                                      │         │  │
│  │  └──────────────────────────┬───────────────────────────┘         │  │
│  │                             │                                     │  │
│  │                             ▼                                     │  │
│  │  ┌──────────────────────────────────────────────────────┐         │  │
│  │  │                   DROPOUT                            │         │  │
│  │  │                                                      │         │  │
│  │  │  attn_weights = dropout(attn_weights)                │         │  │
│  │  │  (B, H, L, S)                                        │         │  │
│  │  └──────────────────────────┬───────────────────────────┘         │  │
│  │                             │                                     │  │
│  │                             │                                     │  │
│  │                             ▼                                     │  │
│  │  ┌──────────────────────────────────────────────────────┐         │  │
│  │  │             APPLY ATTENTION TO VALUES                │         │  │
│  │  │                                                      │         │  │
│  │  │  attn_output = matmul(attn_weights, V_heads)         │         │  │
│  │  │                                                      │         │  │
│  │  │  (B, H, L, D_h)                                      │         │  │
│  │  └──────────────────────────┬───────────────────────────┘         │  │
│  └─────────────────────────────┼─────────────────────────────────────┘  │
│                                │                                        │
│                                ▼                                        │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                    OUTPUT PROJECTION                             │   │
│  │                                                                  │   │
│  │  ┌──────────────────────────────────────────────────────┐        │   │
│  │  │            RESHAPE FROM MULTIPLE HEADS                │       │   │
│  │  │                                                      │        │   │
│  │  │  attn_output = reshape_and_concat_heads(attn_output) │        │   │
│  │  │                                                      │        │   │
│  │  │  (S, B, E)                                           │        │   │
│  │  └──────────────────────────┬───────────────────────────┘        │   │
│  │                             │                                    │   │
│  │                             ▼                                    │   │
│  │  ┌──────────────────────────────────────────────────────┐        │   │
│  │  │               OUTPUT LINEAR LAYER                    │        │   │
│  │  │                                                      │        │   │
│  │  │  attn_output = out_proj(attn_output)                 │        │   │
│  │  │                                                      │        │   │
│  │  │  (S, B, E)                                           │        │   │
│  │  └──────────────────────────┬───────────────────────────┘        │   │
│  └─────────────────────────────┼────────────────────────────────────┘   │
│                                │                                        │
└────────────────────────────────┼────────────────────────────────────────┘
                                 │
                                 ▼
                          ┌─────────────────┐
                          │   FINAL OUTPUT  │
                          │    (S, B, E)    │
                          └─────────────────┘
"""