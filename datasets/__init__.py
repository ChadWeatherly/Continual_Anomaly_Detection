import os
import random
from torch.utils.data import Dataset
from torchvision.io import read_image
from .mvtec import mvtec
from .mtd import mtd

def create_mvtec_supervised():
    """
    On average, we have about 17 images per anomaly type in each category.
    In practice, we don't often have access to that many anomalous cases,
    so let's use 20% for testing and 80% for training.
    """
    train_frac = 0.2
    path = "datasets/mvtec_anomaly_detection/supervised/"
    for cat in os.listdir(path): # Iterate through categories
        if "." not in cat:
            for anom_class in os.listdir(f"{path}{cat}/test"): # Iterate through category anomaly types in testing set
                if anom_class != "good":
                    for img_file in os.listdir(f"{path}{cat}/test/{anom_class}"): # Iterate through each image
                        if img_file.endswith(".png"):
                            if random.random() <= train_frac:
                                old_path = f"{path}{cat}/test/{anom_class}"
                                new_path = f"{path}{cat}/train/{anom_class}"
                                if not os.path.isdir(new_path):
                                    os.makedirs(new_path, 0o777)
                                    print('made dir', sep="\n")
                                os.rename(f"{old_path}/{img_file}", f"{new_path}/{img_file}")

    return

def create_mtd_supervised():
    """
    We are just doing a regular 70/30 split for training/testing
    """
    train_frac = 0.7
    root = 'datasets/magnetic_tile_defects/supervised'
    for folder in os.listdir(root): # Iterate through folders in supervised
        if "." not in folder:
            path = f'{root}/{folder}'
            os.chmod(path, 0o777)
            if folder.startswith('MT_'): # Indicates it's data
                # Let's first add the folder name to the train/test set if they don't exist
                if not os.path.isdir(f'{root}/train/{folder}'):
                    os.mkdir(f'{root}/train/{folder}', 0o777)
                if not os.path.isdir(f'{root}/test/{folder}'):
                    os.mkdir(f'{root}/test/{folder}', 0o777)
                # Next, let's iterate and sort images
                img_path = f'{path}/Imgs' # So img_path is datasets/magnetic_tile_defects/supervised/MT_XXX/Imgs
                os.chmod(img_path, 0o777)
                for img in os.listdir(img_path):
                    if img.endswith(".jpg"): # Get only real images (jpg); masks are saved with same name but as .png
                        # Sort by train/test
                        img_name =img.split(".")[0]
                        if random.random() <= train_frac:
                            os.rename(f"{img_path}/{img}", f"{root}/train/{folder}/{img}")
                            os.rename(f"{img_path}/{img_name}.png", f"{root}/train/{folder}/{img_name}.png")
                        else:
                            os.rename(f"{img_path}/{img}", f"{root}/test/{folder}/{img}")
                            os.rename(f"{img_path}/{img_name}.png", f"{root}/test/{folder}/{img_name}.png")
    return

def create_mtd_unsupervised():
    """
    We are moviing over 50% of all normal images (35% of dataset) to the train set
    and the rest to the test set.
    """
    train_frac = 0.5
    root = 'datasets/magnetic_tile_defects/unsupervised'
    if not os.path.isdir(f'{root}/train/MT_Free'):
        os.mkdir(f'{root}/train/MT_Free', 0o777)
    for folder in os.listdir(root): # Iterate through folders in supervised
        if "." not in folder:
            path = f'{root}/{folder}'
            os.chmod(path, 0o777)
            if folder.startswith('MT_'): # Indicates it's data
                # Let's first add the folder name to the train/test set if they don't exist
                if not os.path.isdir(f'{root}/test/{folder}'):
                    os.mkdir(f'{root}/test/{folder}', 0o777)
                # Next, let's iterate and sort images
                img_path = f'{path}/Imgs' # So img_path is datasets/magnetic_tile_defects/supervised/MT_XXX/Imgs
                os.chmod(img_path, 0o777)
                for img in os.listdir(img_path):
                    if img.endswith(".jpg"): # Get only real images (jpg); masks are saved with same name but as .png
                        # Sort by train/test
                        img_name =img.split(".")[0]
                        if random.random() <= train_frac and folder == 'MT_Free':
                            os.rename(f"{img_path}/{img}", f"{root}/train/{folder}/{img}")
                            os.rename(f"{img_path}/{img_name}.png", f"{root}/train/{folder}/{img_name}.png")
                        else:
                            os.rename(f"{img_path}/{img}", f"{root}/test/{folder}/{img}")
                            os.rename(f"{img_path}/{img_name}.png", f"{root}/test/{folder}/{img_name}.png")
    return