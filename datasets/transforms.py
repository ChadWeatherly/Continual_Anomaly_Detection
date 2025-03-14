"""
Code to perform our transformations for the MTD dataset given for the continual learning task.
"""

from datasets import *

# Set random seed for reproducibility
random.seed(42)

class color_transform():
    """
    Applies a color transform to the image.
    Args:
        img: Image to be transformed
        window: Tuple of window values to be used for the color transform
    Returns:
        Transformed image
    """
    def __init__(self, window):
        self.value = random.uniform(window[0], window[1])
        self.value = random.uniform(1-self.value, 1+self.value)
    def __call__(self, img):
        # Apply color jitter
        img = F.adjust_brightness(img, self.value)
        img = F.adjust_contrast(img, self.value)
        img = F.adjust_saturation(img, self.value)

        return img

class geometric_transform():
    """
    Applies a geometric transform to the image.
    Args:
        img: Image to be transformed
        degrees: +/- degree range to rotate the image
        translate: Tuple of translation values to be used for the transform
        scale: Scale factor
        shear: Shear factor
    Returns:
        Transformed image
    """
    def __init__(self, degrees, translate, scale, shear):
        # Will give us a rotation of +/- degrees
        self.degrees = random.uniform(-degrees, degrees)
        # Need two random numbers, one for vertical and one for horizontal
        self.translate = [random.uniform(0, translate),
                          random.uniform(0, translate)]
        self.scale = random.uniform(1 - scale, 1 + scale)
        self.shear = random.uniform(-shear, shear)

    def __call__(self, img):
        return F.affine(img,
                        self.degrees,
                        self.translate,
                        self.scale,
                        self.shear)
