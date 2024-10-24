import numpy as np
import torch
import torch.nn as nn
from torchvision.models.vision_transformer import vit_b_16
from torchvision.models import ViT_B_16_Weights
from torch.distributions.normal import Normal
import torch.nn.functional as F


class DNE(nn.Module):  # Distribution of Normal Embeddings
    def __init__(self):
        super().__init__()  # Make sure to inherit all methods/properties from nn.Module
        # Get cpu, gpu or mps device for training.
        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        print(f"Using {self.device} device")
        # Get pre-trained Vision Transformer (ViT-B_16, like paper)
        self.vit = vit_b_16(weights='DEFAULT')  # Pre-trained weights on ImageNet
        # head is its own module, so we can easily freeze layers in autograd
        self.head = nn.Sequential(
            nn.Linear(in_features=768, out_features=2),
            nn.Softmax(dim=1)
        )
        # Creating memory mechanism
        # Total memory, M, which will contain a list of statistics (mean, cov, num) for each task
        self.memory = []
        # z_epoch holds embeddings for samples in  last epoch (epoch 50 in paper)
        self.z_epoch = None  # num_samples x 768
        # all samples taken during inference time from task distributions
        self.z_global = None
        # move to best device
        self.to(self.device)
        return

    def forward(self, img, head=False, add_to_z_epoch=False):
        # Calculate embeddings, Z, from ViT
        embeds = self.embed(img)  # B x 768
        # Add embeddings, Z, to z_epoch
        if add_to_z_epoch:  # The paper only uses the last epoch's data, so why calculate each epoch?
            z = embeds.detach().clone()
            if self.z_epoch is None:
                self.z_epoch = z
            else:
                self.z_epoch = torch.cat((self.z_epoch, z))
        # Only pass embeddings through head if training model
        if head:
            preds = self.head(embeds)
            return preds

        return embeds

    def embed(self, img):
        """
        Gets the final feature embeddings of the ViT-B_16 model, before passing to the head for classification.
        Each of these layers was checked against the original ViT paper, code implementation, and current paper
        - Takes in any image of whatever size and does transformation
        - Outputs the final class-token, which is of size (batch_size B, 768 (Latent dimension size=768 per token/patch))
        """
        # Creates patch_embeddings based on transformed image
        img = img.unsqueeze(0) if len(
            img.shape) == 3 else img  # Make sure we have a batch dimension if single, unbatched image was given
        patch_embeds = self.vit._process_input(img)  # B x 196 x 768 (196 patches/tokens, latent dimension D = 768)
        # Expand class token to the full batch, size B x 1 x 768
        batch_class_token = self.vit.class_token.expand(img.shape[0], -1, -1)
        # Add class token to patch embeddings
        patch_embeds = torch.cat([batch_class_token, patch_embeds], dim=1)  # B x 197 x 768
        # Pass the patch embeddings through the transformer encoder
        features = self.vit.encoder(patch_embeds)  # B x 197 x 768
        # We only want the final output features of the class token, so a 1D vector of size D (768) for each img
        cls_features = features[:, 0]  # B x 768
        return cls_features

    def freeze_head(self, freeze):
        # freeze is either True/False, whether to freeze/unfreeze the head layer module parameters (won't update)
        for param in self.head.parameters():
            param.requires_grad = not freeze
        return

    def update_memory(self):
        """
        Updates the long-term memory based on z_epoch,
        which after epoch 50 should be added into memory

        z_epoch should be of size (num_task, 768), where
            num_task is the number of samples in the current task
            768 is the ViT latent dimension size, D
        """
        task_memory = []  # will contain num_task, mean, covariance
        # Add num_task
        task_memory.append(self.z_epoch.shape[0])
        # Add mean matrix, should be a 1D tensor of length 768
        task_memory.append(self.z_epoch.mean(dim=0))
        # Add standard deviation
        # Paper does covariance matrix, but we need standard deviation to create
        # and sample from distribution in inference. After the distribution is created,
        # we can sample for global distribution and calculate covariance matrix, when it's really needed
        task_memory.append(self.z_epoch.std(dim=0))

        self.z_epoch = None
        self.memory.append(task_memory)
        return

    def generate_global_samples(self):
        """
        Based on the paper, a global mean and covariance distribution sampling is generated from the memory.
        We have to generate the samples, because the sklearn ShrunkCovariance class has to be used on cpu,
            which may conflict with model being on GPU or MPS.
        """
        self.z_global = None
        # we want to re-generate our original dataset from the memory distributions
        for t in self.memory:
            n_t, mu, sigma = t
            gaussian = Normal(mu, sigma)
            task_samples = gaussian.sample((n_t,))  # samples n_t samples from task guassian
            if self.z_global is None:
                self.z_global = task_samples
            else:
                self.z_global = torch.cat((self.z_global, task_samples))

        return self.z_global
