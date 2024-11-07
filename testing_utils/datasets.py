from torch.utils.data import Dataset
import torch
import os
import numpy as np
import random
from torchvision.io import read_image
import torchvision.transforms.v2 as transforms
from torchvision.models import ViT_B_16_Weights


class mvtec(Dataset):
    def __init__(self, multi=True, mtd=False, train=True, task=None, cutpaste=False):
        """
        Creates MVTEC dataset for use in PyTorch. Based on Towards Continaul Anomaly Detection paper
        Args:
            multi: Whether to do multi-category or not
            mtd: Whether to add Magnetic Tile Defects or not
            train: Whether the dataset is used for training or testing
            task: Which task, an int in [1, 5]
            cutpaste: Whether to do cutpaste transformation or not (found in their original code)
            cutpaste:
        """
        # Don't need to transform; ViT does that
        self.multi = multi  # Whether to do multi-category or not
        self.mtd = mtd  # Whether to add in Magnetic Tile Defects dataset
        self.train = train  # Whether this is the training or testing set
        self.task = task  # Which task, should be a number 1-5
        self.mvtec_path = "mvtec_anomaly_detection/"
        self.categories = None
        self.get_task_categories()  # creates self.categories, containing all needed category labels in a list
        self.filenames = None
        self.labels = None  # 1 = anomaly, 0 = good
        self.get_all_filenames()  # creates list of all filenames and paths
        self.cutpaste = cutpaste  # Whether to use cutpaste transforms or not
        vit_transforms = ViT_B_16_Weights.DEFAULT.transforms()  # Pre-processing transforms for ViT (resize, normalize)
        self.pre_transforms = transforms.Compose([
            vit_transforms,
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        ])
        self.post_transforms = transforms.Compose([
            transforms.RandomRotation(90),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        if cutpaste:
            # These transforms are copied from the original code, which does them sneakily without telling you
            self.cutpaste_transform = CutPaste()

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
        if self.cutpaste:
            return len(self.filenames) * 2
        else:
            return len(self.filenames)

    def __getitem__(self, idx):
        # Images need to be pre-processed beforehand so dataloader handles same size images
        # need to do is make sure the image has 3 channels
        if self.cutpaste:
            label = 1 if idx >= len(self.filenames) else 0
            new_idx = idx % len(self.filenames)
            category = self.filenames[new_idx].split('/')[1]
            img = read_image(self.filenames[new_idx]).expand(3, -1, -1)
            # Transforms the image from 3xHxW to 3x224x224 and normalizes all values to [0, 1]
            img = self.pre_transforms(img)  # B x 3 x 224 x 224
            if label == 1:
                img = self.cutpaste_transform(img)
            img = self.post_transforms(img)
            return img, label, category
        else:
            category = self.filenames[idx].split('/')[1]
            img = read_image(self.filenames[idx]).expand(3, -1, -1)
            # Transforms the image from 3xHxW to 3x224x224 and normalizes all values to [0, 1]
            img = self.post_transforms(self.pre_transforms(img))  # B x 3 x 224 x 224
            return img, self.labels[idx], category


class CutPaste(object):

    def __init__(self):
        '''
        This class creates to different augmentation CutPaste and CutPaste-Scar. Moreover, it returns augmented images
        for binary and 3 way classification

        :arg
        :transform[binary]: - if True use Color Jitter augmentations for patches
        '''
        self.to_pil = transforms.ToPILImage()
        self.to_tensor = transforms.Compose([
            transforms.ToImage(),
            transforms.ToDtype(torch.float32)
        ])

    @staticmethod
    def crop_and_paste_patch(image, patch_w, patch_h, rotation=False):
        """
        Crop patch from original image and paste it randomly on the same image.

        :image: [PIL] _ original image
        :patch_w: [int] _ width of the patch
        :patch_h: [int] _ height of the patch
        :transform: [binary] _ if True use Color Jitter augmentation
        :rotation: [binary[ _ if True randomly rotates image from (-45, 45) range

        :return: augmented image
        """

        org_w, org_h = image.size
        mask = None

        patch_left, patch_top = random.randint(0, org_w - patch_w), random.randint(0, org_h - patch_h)
        patch_right, patch_bottom = patch_left + patch_w, patch_top + patch_h
        patch = image.crop((patch_left, patch_top, patch_right, patch_bottom))

        if rotation:
            random_rotate = random.uniform(*rotation)
            patch = patch.rotate(random_rotate, expand=True)
            mask = patch.split()[-1]

        # new location
        paste_left, paste_top = random.randint(0, org_w - patch_w), random.randint(0, org_h - patch_h)
        aug_image = image.copy()
        aug_image.paste(patch, (paste_left, paste_top), mask=mask)
        return aug_image

    def cutpaste(self, image, area_ratio=(0.02, 0.15), aspect_ratio=((0.3, 1), (1, 3.3))):
        '''
        CutPaste augmentation

        :image: [PIL] - original image
        :area_ratio: [tuple] - range for area ratio for patch
        :aspect_ratio: [tuple] -  range for aspect ratio

        :return: PIL image after CutPaste transformation
        '''

        img_area = image.size[0] * image.size[1]
        patch_area = random.uniform(*area_ratio) * img_area
        patch_aspect = random.choice([random.uniform(*aspect_ratio[0]), random.uniform(*aspect_ratio[1])])
        patch_w = int(np.sqrt(patch_area * patch_aspect))
        patch_h = int(np.sqrt(patch_area / patch_aspect))
        cutpaste = self.crop_and_paste_patch(image, patch_w, patch_h, rotation=False)
        return cutpaste

    def __call__(self, image):
        '''

        :image: [torch.tensor] - original image
        :return: if type == 'binary' returns original image and randomly chosen transformation, else it returns
                original image, an image after CutPaste transformation and an image after CutPaste-Scar transformation
        '''
        image = self.to_pil(image)
        cutpaste = self.cutpaste(image)
        cutpaste = self.to_tensor(cutpaste)
        return cutpaste
