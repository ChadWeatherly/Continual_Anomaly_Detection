# __init__.py is called when imported by another Python file, where this directory is a python package, consisting modules of .py files

# Setting __all__ tells Python which modules (.py files) to import when importing this package
# __all__ = ['dne']

import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
import numpy as np


def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
