from torch.utils.data import Dataset
import os
from torchvision.io import read_image


class mvtec(Dataset):
    def __init__(self, train=True, task=None):
        """
        Creates MVTEC dataset for use in PyTorch. Based on Towards Continaul Anomaly Detection paper
        Args:
            train: Whether the dataset is used for training or testing
            task: Which task, a string of one of the 12 mvtec categories
        """
        self.train = train  # Whether this is the training or testing set
        self.task = task  # Which task, should be a number 1-5
        self.mvtec_path = f"datasets/mvtec_anomaly_detection/{task}"
        self.filenames = []
        self.labels = []  # 1 = anomaly, 0 = good
        self.get_all_filenames()  # creates list of all filenames and paths, self.filenames
        # TODO: Define transforms

        return

    def get_all_filenames(self):
        """
        creates list of all image filenames, a list of strings
        """
        if self.train:
            cat_path = f'{self.mvtec_path}/train/good/'
            for file in os.listdir(cat_path):
                if file.endswith('.png'):
                    self.filenames.append(f'{cat_path}{file}')
                    self.labels.append(0)
        else:
            cat_path = f'{self.mvtec_path}/test/'
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
        # Images need to be pre-processed beforehand so dataloader handles same size images
        # need to do is make sure the image has 3 channels
        img = read_image(self.filenames[idx]).expand(3, -1, -1)

        return img, self.labels[idx]

class mtd(Dataset)
    def __init__(self, train=True, task=None):
        return

    def __getitem__(self, idx):
        return

    def __len__(self):
        return