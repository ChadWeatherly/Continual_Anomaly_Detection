import os
import random
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.utils import make_grid
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from .mvtec import mvtec
from .mtd import mtd

### Show a list of tensor images
def show(imgs):
    imgs = make_grid(imgs)
    imgs = imgs.unsqueeze(0) if imgs.dim() == 3 else imgs
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    fig.set_size_inches(15, 12)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img), cmap="gray")
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])