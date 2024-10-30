from torchvision import transforms
from .transforms import aug_transformation, no_aug_transformation
from .mvtec_dataset import MVTecAD
from .revdis_mvtec_dataset import RevDisTestMVTecDataset
from torch.utils.data import DataLoader
from .utils import get_mvtec_classes


def get_mvtec_dataloaders(args, t, dataloaders_train, dataloaders_test, learned_tasks, all_test_filenames):
    """
    Creates dataloaders for sequential/continual learning on MVTec dataset.

    Args:
        args: Configuration arguments
        t (int): Current task index
        dataloaders_train (list): List of training dataloaders for previous tasks
        dataloaders_test (list): List of testing dataloaders for previous tasks
        learned_tasks (list): List of previously learned product categories
        all_test_filenames (list): List of test file paths from previous tasks

    Returns:
        Tuple containing:
        - Current task's training dataloader
        - Updated list of all training dataloaders
        - Updated list of all test dataloaders
        - Updated list of learned tasks
        - Number of training samples in current task
        - Updated list of all test filenames
        - Current task's training dataset
        - Current task's test dataset
    """
    # Get ordered list of MVTec product categories based on specified order in args
    mvtec_classes = get_mvtec_classes(args)

    # Get number of classes to learn per task
    N_CLASSES_PER_TASK = args.dataset.n_classes_per_task

    # Handle different incremental learning settings
    if args.dataset.data_incre_setting == 'one':
        # Single-category incremental learning:
        # First task learns 10 categories, then 1 new category per subsequent task
        if t == 0:
            task_mvtec_classes = mvtec_classes[: 10]  # First 10 classes
        else:
            i = 10 + (t - 1) * N_CLASSES_PER_TASK  # Start index for current task
            task_mvtec_classes = mvtec_classes[i: i + N_CLASSES_PER_TASK]
    else:
        # Multi-category incremental learning:
        # Learn N_CLASSES_PER_TASK categories in each task
        i = t * N_CLASSES_PER_TASK
        task_mvtec_classes = mvtec_classes[i: i + N_CLASSES_PER_TASK]

    # Keep track of categories learned in this task
    learned_tasks.append(task_mvtec_classes)

    # Get data transformations
    train_transform = aug_transformation(args)  # Includes augmentations for training
    test_transform = no_aug_transformation(args)  # Basic transforms for testing

    # Create datasets based on model type
    if args.model.method == 'revdis':
        # Special case for Reverse Distillation method
        train_data = MVTecAD(args.data_dir, task_mvtec_classes,
                             transform=test_transform, size=args.dataset.image_size)
        test_data = RevDisTestMVTecDataset(args.data_dir, task_mvtec_classes,
                                           size=args.dataset.image_size)
        all_test_filenames.append(test_data.img_paths)
    else:
        # Standard case for other methods
        train_data = MVTecAD(args.data_dir, task_mvtec_classes,
                             transform=train_transform, size=args.dataset.image_size)
        test_data = MVTecAD(args.data_dir, task_mvtec_classes, args.dataset.image_size,
                            transform=test_transform, mode="test")
        all_test_filenames.append(test_data.all_image_names)

    # Create dataloaders
    train_dataloader = DataLoader(
        train_data,
        batch_size=args.train.batch_size,
        shuffle=True,
        num_workers=args.dataset.num_workers
    )
    dataloaders_train.append(train_dataloader)

    dataloader_test = DataLoader(
        test_data,
        batch_size=args.eval.batch_size,
        shuffle=False,
        num_workers=args.dataset.num_workers
    )
    dataloaders_test.append(dataloader_test)

    # Print task information
    print('class name:', task_mvtec_classes,
          'number of training sets:', len(train_data),
          'number of testing sets:', len(test_data))

    return (train_dataloader, dataloaders_train, dataloaders_test,
            learned_tasks, len(train_data), all_test_filenames,
            train_data, test_data)
