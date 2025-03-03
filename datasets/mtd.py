"""
This file contains the mtd dataset class, which contains only one type of object, magnetic tiles.

There are 1344 images in total, with ~70% of those images being normal.
The rest of the images are split into 5 anomalous type classes:
    - Blowhole
    - Break
    - Crack
    - Fray
    - Uneven

Each image (.jpg) has a corresponding ground truth mask file (.png).

For both unsupervised and supervised experiments, we want to have about a 70/30 training/testing split.
TODO: Think about training/testing splits between unsupervised and supervised.
In unsupervised, only 70% of normal images will be moved to the training set.
In supervised, 70% of all images will be moved to the training set.
"""

from torch.utils.data import Dataset
import os
from torchvision.io import read_image

class mtd(Dataset):
    def __init__(self, train=True, task=None, unsupervised=True):
        """
        Creates MVTEC dataset for use in PyTorch
        Args:
            train: Whether the dataset is used for training or testing, during training, only normal samples are seen
            task: Which task, a string of one of the 12 mvtec categories
            unsupervised: Whether to use unsupervised or supervised training, i.e. whether training uses only normal
                            samples or not
        """
        self.train = train  # Whether this is the training or testing set
        self.task = task
        self.unsupervised = unsupervised

        # Path is the general path to the training/testing set of a given task
        # We need the path to also extract groud truth (gt) images
        self.path = ('datasets/mvtec_anomaly_detection/' +
                     f'{'unsupervised/' if unsupervised else 'supervised/'}' +
                     f'{task}/')
        # print(self.path)

        self.filenames = []
        self.labels = []  # 1 = anomaly, 0 = good
        self.get_all_filenames()  # creates list of all filenames and paths, self.filenames
        return

    def get_all_filenames(self):
        """
        creates list of all image filenames, a list of strings
        """
        img_path = self.path + 'train' if self.train else 'test'
        for anom_type in os.listdir(img_path): # iterating through anomaly types
            if "." not in anom_type: # Making sure the folder is not a file
                for img in os.listdir(f'{img_path}/{anom_type}'): # iterating through images
                    if img.endswith(".png"):
                        self.filenames.append(f"{img_path}/{anom_type}/{img}")
                        self.labels.append(1 if anom_type != "good" else 0)

        return

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        # Images need to be pre-processed beforehand so dataloader handles same size images
        # need to do is make sure the image has 3 channels
        img_filename = self.filenames[idx]
        img = read_image(img_filename)#.expand(3, -1, -1)

        # Get ground truth image
        img_split = img_filename.split('/')
        anom_type = img_split[-2]
        if anom_type == 'good':
            gt_img = None
        else:
            img_num = img_split[-1].split('.')[0]
            gt_filename = f'{self.path}ground_truth/{anom_type}/{img_num}_mask.png'
            print(gt_filename)
            gt_img = read_image(gt_filename)

        return {'image': img,
                'label': self.labels[idx],
                'ground_truth': gt_img}