"""
Our dataset classes for MVTec-AD and Magnetic Tile Defects.

MVTEC-AD: https://www.mvtec.com/company/research/datasets/mvtec-ad
MVTec-AD contains 15 categories of images:
    - bottle
    - cable
    - capsule
    - carpet
    - grid
    - hazelnut
    - leather
    - metal_nut
    - pill
    - screw
    - tile
    - transistor
    - wood
    - zipper

We differentiate datasets by whether they are
- training or testing
- which task, a string of one of the 12 mvtec categories
- whether it is unsupervised or supervised, i.e. whether training uses only normal samples or not

The unsupervised dataset is the same as the original MVTEC-AD dataset, which already splits each category into a
training and testing set, but there are only normal ("good") samples in the training set.

In order to create the supervised dataset, we first assume that during real-life supervised scenarios, it's possible
that there are some anomalies, but not many. There are already good examples in the training and testing sets, but we
move ~20% of the anomalies from the testing set to the training set.
"""

from torch.utils.data import Dataset
import os
from torchvision.io import read_image


# TODO: Add transforms if necessary
class mvtec(Dataset):
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
        img_num = img_split[-1].split('.')[0]
        gt_filename = f'{self.path}ground_truth/{anom_type}/{img_num}_mask.png'
        print(gt_filename)
        gt_img = read_image(gt_filename)

        return {'image': img,
                'label': self.labels[idx],
                'ground_truth': gt_img}

class mtd(Dataset):
    def __init__(self, train=True, task=None):
        return

    def __getitem__(self, idx):
        return

    def __len__(self):
        return