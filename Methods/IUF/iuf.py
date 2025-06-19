# Methods/IUF/iuf.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision.models.vision_transformer import vit_b_16
from Methods import BaseAnomalyDetector
from Methods.IUF.utils.discriminator import Discriminator
from Methods.IUF.utils.encoder import Encoder
from Methods.IUF.utils.decoder import Decoder

class IUF_Model(BaseAnomalyDetector):
    """
    Complete IUF Module

    We have one ViT class that we build on,
    From there, we will add methods to create the discriminator, encoder, and decoder.

    Algorithm Notes:

        ViT
        - Added positional encoding to the tokens
        - Changed BatchNorm to LayerNorm and ReLU to GELU, in accordance with the ViT paper
        - I noticed that in their Multi-head attention, the authors only work on row-wise attention to simplify their computations, so I am going to operate on patches, as the multiplication is too large.
        - Batch size needs to maybe be larger than the embedding dimension?
    """
    def __init__(self,
                 in_channels=3,
                 patch_size=16, # 224 should be divisible by patch_size
                 embed_dim=64,
                 num_heads=4, # Should be able to take the sqrt easily
                 num_layers=4,
                 num_tasks=15):
        super().__init__()

        self.embed_dim = embed_dim

        # Discriminator is responsible for predicting the task of a given image, and
        # its intermediary features are combined in the encoder with the query to get
        # "Object-Aware Self Attention" (OASA) modules in the encoder
        self.discriminator = Discriminator(device=self.device,
                                           in_channels=in_channels,
                                           patch_size=patch_size,
                                           embed_dim=embed_dim,
                                           num_heads=num_heads,
                                           num_layers=num_layers,
                                           output_size=num_tasks)
        # Encoder creates our latent space and performs SVD on it,
        # which is used in the gradient update
        self.encoder = Encoder(device=self.device,
                               in_channels=in_channels,
                               patch_size=patch_size,
                               embed_dim=embed_dim,
                               num_heads=num_heads,
                               num_layers=num_layers)
        # The decoder decodes the latent space into a reconstructed image
        self.decoder = Decoder(device=self.device,
                               in_channels=in_channels,
                               patch_size=patch_size,
                               embed_dim=embed_dim,
                               num_heads=num_heads,
                               num_layers=num_layers)

        # The previous grad values are used for regularizing the gradient update
        self.prev_grad = {}
        self.v_old = None
        # This multiplier is the same for all gradient updates
        self.omega = torch.arange(1, 65).log().to(self.device)

        return

    def update_grad(self, loss, beta=0.5):
        """
        Method to calculate the regularized gradient

        Args:
            - loss: the calculated loss value
            - beta: the beta hyperparameter

        Returns:
        """
        # First, call loss.backward() to calculate the gradient for all params
        loss.backward()

        # Iterate through parameters and adjust their gradients
        for p in self.named_parameters():
            name = p[0]
            param = p[1] # Updates to the parameter tensor happen in place

            # Some gradients don't get computed, so just move on
            if param.grad is None:
                continue
            # On the first iteration, we won't have an old set of gradients, so just move on
            elif name not in self.prev_grad.keys():
                self.prev_grad[name] = param.grad.clone()
            elif param.shape[-1]==self.embed_dim:
                # Only do something if the parameter has the correct dimension

                try:
                    # Calculate grad_star, @ = classic matrix multiplication
                    t1 = self.v_old @ param.grad
                    t2 = self.omega * t1
                    grad_star = self.v_old.inverse() @ t2
                    # Create new gradient update
                    param.grad = grad_star + (beta * self.prev_grad[name])
                except:
                    pass
            else:
                continue

        return

    def train_one_epoch(self, dataloader, optimizer, task_num, **kwargs):
        """
        Train a model on one epoch.
        Args:
            dataloader: Dataloader for the training data.
            optimizer: torch.optim.Optimizer
            task_num: Which task is being trained on, starting at 1
            kwargs: Additional keyword arguments

        Returns: epoch_loss, the total accumulated loss for that epoch

        """
        self.train()

        epoch_loss = 0
        for batch_idx, data in enumerate(dataloader):
            optimizer.zero_grad()
            imgs = data['image'].to(self.device)
            x_recon, d_out, s_vals = self.forward(imgs)
            batch_size = imgs.shape[0]
            loss = IUF_Loss(x=imgs,
                            x_recon=x_recon,
                            singular_vals=s_vals,
                            t=3,
                            discrim_output=d_out,
                            task_idx=(torch.ones(batch_size, dtype=torch.long) * (task_num-1)).to(self.device)
                            )
            self.update_grad(loss)
            epoch_loss += loss.item()
            optimizer.step()

        return epoch_loss

    def eval_one_epoch(self, dataloader, results_path=None, model_path=None):
        """
        Eval a model on one epoch.
        Args:
            dataloader: Dataloader for the training data.
            results_path: If exists, where we want to save data about the results
            model_path: If exists, where we want to access model params

        Returns:
            predictions: the predicted labels
            labels: the ground truth labels
        """
        self.eval()

        preds = []
        all_labels = []
        for batch_idx, data in enumerate(dataloader):
            imgs = data['image'].to(self.device)

            # Get epoch loss
            logits = self.forward(imgs, head=True,
                                  add_to_z_epoch=False).detach().clone()
            logits = F.softmax(logits, dim=1).argmax(dim=1).detach().cpu()
            preds += [i.item() for i in logits]
            all_labels += data['label']


        # Note: each of these values are just float values (taken from Tensor.item())
        return preds, all_labels

    def forward(self, x):
        x.to(self.device)

        # oasa_features = list of length num_layers,
        # where each item is a tensor of size (B x L x E)

        # d_out is of size (B x num_classes)
        oasa_features, d_out = self.discriminator(x, return_features=True)

        z, u, s, v = self.encoder(x, oasa_features)
        # z = latent features, (B x L x E)
        # u, s, v from SVD
        # u, (B x B)
        # S, (B | C), whichever is smaller
        # V, (C x C), C = channels/embed_dim

        x_recon = self.decoder(z)

        # v_old is used during the gradient update
        self.v_old = v.clone()

        return x_recon, d_out, s

    def save(self, model_path):
        """
        Saves the model to disk
        Args:
            model_path: a string of the model path
        Returns:
        """
        # Save model dict, based on BaseAnomalyDetector
        super().save(path=model_path)

        return

    def load(self, model_path):
        """
        Loads a model from a file
        Args:
            model_path: a string of the model path

        Returns:

        """
        # Load model dict, based on BaseAnomalyDetector
        super().load(path=model_path)

        return

def IUF_Loss(x, x_recon, singular_vals, t, discrim_output, task_idx,
             lamdba1=1.0, lambda2=0.5, lambda3=8):
    """
    Loss function consists of the following components:
        - Reconstruction error = abs(x_recon - x)
        - Discriminator error = CrossEntropy(discrim_output, true label)
        - Singular Value error = sum(singular_vals from t -> C), from SVD(M_hat),
                                where t is a hyperparameter and C is the total number of singular values.
        Each component is weighted by a corresponding lambda, where by default from the paper
        - lambda1 = 1
        - lambda2 = 0.5
        - lambda3 = 1-10

    Args:
        x: true img
        x_recon: reconstructed img
        singular_vals: Torch.Tensor(B | C), singular values from SVD of latent space
        t: int, a hyperparameter to control how many of the singular values are summed up,
                sum(singular_vals[t:len(singular_vals)]),
                where t=0 indicates that all singular values are regulated. t is equal
                to the number of top singular values we want to exclude from regularization
        discrim_output: Torch.Tensor(B, num_classes), output of discriminator
        task_idx: Torch.Tensor(B), true task indices
        lamdba1, lambda2, lambda3: weight hyperparams for 3 loss components

    Returns: Final IUF loss, based on given lambda parameters
    """
    assert t < len(singular_vals), "t should be less than the number of singular values"

    ### Reconstruction Error
    recon_error = torch.abs(x_recon - x).mean()
    # print("Reconstruction error: ", recon_error.item())

    ### Discriminator Error
    d_err = F.cross_entropy(discrim_output, task_idx, reduction='sum')
    # print("Discriminator error: ", d_err.item())

    ### Singular Value Error
    s_val_error = singular_vals[t:].sum()
    # print("Singular Value error: ", s_val_error.item())

    return (lamdba1 * recon_error) + \
        (lambda2 * d_err) + \
        (lambda3 * s_val_error)

