from collections.abc import Iterable
from pathlib import Path
from PIL import Image
from joblib import Parallel, delayed
from torch.utils.data import Dataset


def flatten(items, ignore_types=(str, bytes)):
    """
    Recursively flattens nested iterables (e.g., lists of lists) into a single list.
    Args:
        items: The nested iterable to flatten
        ignore_types: Types to not flatten (str and bytes by default)
    Returns:
        Generator yielding flattened items
    """
    for x in items:
        if isinstance(x, Iterable) and not isinstance(x, ignore_types):
            yield from flatten(x)
        else:
            yield x


class Repeat(Dataset):
    """
    A wrapper dataset that repeats the original dataset to a new specified length.
    Useful for extending smaller datasets to match larger ones.
    """

    def __init__(self, org_dataset, new_length):
        self.org_dataset = org_dataset
        self.org_length = len(self.org_dataset)
        self.new_length = new_length

    def __len__(self):
        return self.new_length

    def __getitem__(self, idx):
        # Uses modulo to wrap around to the start when reaching the end
        return self.org_dataset[idx % self.org_length]


class MVTecAD(Dataset):
    """
    Dataset class for MVTec Anomaly Detection dataset.
    Handles loading and processing of industrial images for anomaly detection tasks.
    """

    def __init__(self, root_dir, task_mvtec_classes, size, transform=None, mode="train"):
        """
        Initializes the MVTec dataset.
        Args:
            root_dir (string): Root directory containing the MVTec AD dataset
            task_mvtec_classes (list): List of product categories to load (e.g., ['bottle', 'cable'])
            size (int): Size to resize images to
            transform: Transformations to apply to the images
            mode (str): Either "train" (normal samples) or "test" (normal + anomalous samples)
        """
        self.root_dir = Path(root_dir)
        self.task_mvtec_classes = task_mvtec_classes
        self.transform = transform
        self.mode = mode
        self.size = size
        self.all_imgs = []  # Stores preprocessed images for training
        self.all_image_names = []  # Stores paths to all images

        # Load images for each product category
        for class_name in self.task_mvtec_classes:
            if self.mode == "train":
                # For training, only load normal ("good") samples
                self.image_names = list((self.root_dir / class_name / "train" / "good").glob("*.png"))
                self.all_image_names.append(self.image_names)

                print("loading images")
                # Parallel processing to load and resize images
                # Cache images in memory for faster training
                self.imgs = (Parallel(n_jobs=10)(
                    delayed(lambda file: Image.open(file).resize((size, size)).convert("RGB"))(file)
                    for file in self.image_names))
                self.all_imgs.append(self.imgs)
                print(f"loaded {class_name} : {len(self.imgs)} images")
            else:
                # For testing, load all samples (both normal and anomalous)
                # Images are organized in subdirectories by defect type
                self.image_names = list((self.root_dir / class_name / "test").glob(str(Path("*") / "*.png")))
                self.all_image_names.append(self.image_names)

        # Flatten nested lists into single lists
        self.all_imgs, self.all_image_names = list(flatten(self.all_imgs)), list(flatten(self.all_image_names))

    def __len__(self):
        """Returns the total number of images in the dataset"""
        return len(self.all_image_names)

    def __getitem__(self, idx):
        """
        Retrieves an image by index
        Args:
            idx (int): Index of the image to retrieve
        Returns:
            For training: transformed image
            For testing: tuple of (transformed image, anomaly label)
        """
        if self.mode == "train":
            # Return preprocessed training image (only normal samples)
            img = self.all_imgs[idx].copy()
            if self.transform is not None:
                print('train dataset has a transform')
                img = self.transform(img)
            return img
        else:
            # For testing, load image and determine if it's anomalous based on directory name
            filename = self.all_image_names[idx]
            label = filename.parts[-2]  # Directory name indicates defect type
            img = Image.open(filename)
            img = img.resize((self.size, self.size)).convert("RGB")
            if self.transform is not None:
                img = self.transform(img)
            # Return image and binary label (True if defective, False if normal)
            return img, label != "good"