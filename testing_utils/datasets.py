from torch.utils.data import Dataset
import torch
import os
from torchvision.io import read_image
from torchvision.models import ViT_B_16_Weights


class mvtec(Dataset):
    def __init__(self, multi=True, mtd=False, train=True, task=None):
        # Don't need to transform; ViT does that
        self.multi = multi  # Whether to do multi-category or not
        self.mtd = mtd  # Whether to add in Magnetic Tile Defects dataset
        self.train = train  # Whether this is the training or testing set
        self.task = task  # Which task, should be a number 1-5
        self.mvtec_path = "mvtec_anomaly_detection/"
        self.categories = None
        self.get_task_categories()  # creates self.categories, containing all needed categorie labels in a list
        self.filenames = None
        self.labels = None  # 1 = anomaly, 0 = good
        self.get_all_filenames()  # creates list of all filenames and paths
        self.vit_transforms = ViT_B_16_Weights.DEFAULT.transforms()  # Pre-processing transforms
        return

    def get_task_categories(self):
        """
        creates list of task categories for the specified task
        """
        # Get list of all category names
        all_categories = []
        for dir in os.listdir(self.mvtec_path):
            if not dir.endswith('.txt'):
                all_categories.append(dir)
        # split list into task lists
        if self.multi:
            self.categories = [all_categories[i] for i in range(self.task - 1, len(all_categories), 5)]
        else:  # in the case it's single-category (SCIL)
            pass

        return

    def get_all_filenames(self):
        """
        creates list of all image filenames, a list of strings
        """
        self.filenames = []
        self.labels = []
        for cat in self.categories:
            if self.train:
                cat_path = f'{self.mvtec_path}{cat}/train/good/'
                for file in os.listdir(cat_path):
                    if file.endswith('.png'):
                        self.filenames.append(f'{cat_path}{file}')
                        self.labels.append(0)
            else:
                cat_path = f'{self.mvtec_path}{cat}/test/'
                for dir in os.listdir(cat_path):
                    for file in os.listdir(f'{cat_path}{dir}/'):
                        if file.endswith('.png'):
                            self.filenames.append(f'{cat_path}{dir}/{file}')
                            if dir == 'good':
                                self.labels.append(0)
                            else:
                                self.labels.append(1)
        return

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        category = self.filenames[idx].split('/')[1]
        # Images need to be pre-processed beforehand so dataloader handles same size images
        # need to do is make sure the image has 3 channels
        img = read_image(self.filenames[idx]).expand(3, -1, -1)
        # Transforms the image from 3xHxW to 3x224x224 and normalizes all values to [0, 1]
        img = self.vit_transforms(img)  # B x 3 x 224 x 224
        return img, self.labels[idx], category
