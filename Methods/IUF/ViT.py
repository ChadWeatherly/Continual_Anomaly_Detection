"""
Base ViT class for IUF, used to create the
discriminator, encoder, and decoder

ViT Block Diagram at bottom of page
"""

import torch.nn.functional as F
import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights

# TODO: Implement ViT module based on PyTorch's ViT implementation
class ViT(nn.Module):
    def __init__(self, num_classes=15, embed_dim=768):
        super().__init__()

        # Load the pre-trained ViT
        base_model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)

        # Extract the components you want to reuse
        self.patch_embedding = base_model.conv_proj
        self.class_token = base_model.class_token
        self.position_embedding = base_model.encoder.pos_embedding

        # Get the transformer blocks but don't include the final layers
        self.encoder_blocks = base_model.encoder.layers

        # Define your custom components for IUF
        # These would implement your Object-Aware Self-Attention, etc.
        self.discriminator = self._create_discriminator(embed_dim)
        self.decoder = self._create_decoder(embed_dim)

    def _create_discriminator(self, embed_dim):
        """Create the discriminator component of IUF"""
        # Implementation details based on the paper
        return nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.GELU(),
            nn.Linear(512, 15)  # Assuming 15 object classes
        )

    def _create_decoder(self, embed_dim):
        """Create the decoder component of IUF"""
        # This would be your reconstruction pathway
        # Detailed implementation based on the paper
        # ...

    def forward(self, x, category_label=None):
        # Process input through patch embedding
        x = self.patch_embedding(x)

        # Add positional embeddings and class token
        # Similar to original ViT but with your modifications

        # Pass through encoder blocks
        # This is where you would implement OASA

        # Get embeddings for the discriminator
        discriminator_output = self.discriminator(encoder_output)

        # Use the embeddings to generate reconstruction
        # This is where SCL would be applied
        decoder_output = self.decoder(encoder_output)

        return {
            "class_prediction": discriminator_output,
            "reconstruction": decoder_output
        }


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