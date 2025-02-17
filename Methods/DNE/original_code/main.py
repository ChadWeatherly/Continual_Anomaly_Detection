import os
import torch
from tqdm import tqdm
from argument import get_args
from datasets import get_dataloaders
from eval import eval_model
from methods import get_model
from models import get_net_optimizer_scheduler
from utils.density import GaussianDensityTorch
import warnings


def get_inputs_labels(data):
    """Processes input data to handle both single-task and multi-task scenarios

    This function is crucial for handling data augmentation and task transitions:
    - For single task data (normal samples): assigns label 0
    - For multi-task data (augmented/multiple categories): assigns sequential labels

    Args:
        data: Either a single tensor or list of tensors
            - Single tensor: normal samples from current task
            - List of tensors: samples from multiple tasks/augmentations

    Returns:
        tuple: (processed inputs, corresponding labels)
    """
    if isinstance(data, list):
        # Multi-task scenario: each element represents different task/augmentation
        inputs = [x.to(args.device) for x in data]
        # Creates sequential labels (0,1,2...) for each task
        labels = torch.arange(len(inputs), device=args.device)
        labels = labels.repeat_interleave(inputs[0].size(0))
        inputs = torch.cat(inputs, dim=0)
    else:
        # Single task scenario: all data from same task (normal samples)
        inputs = data.to(args.device)
        # All normal samples get label 0
        labels = torch.zeros(inputs.size(0), device=args.device).long()
    return inputs, labels


def main(args):
    """Main training loop implementing the CAD framework

    Key components:
    1. Memory management for task statistics
    2. Periodic evaluation during training
    3. Support for different training methods (panda, upper bound, etc.)
    4. Task-wise distribution tracking
    """
    # Initialize model components
    net, optimizer, scheduler = get_net_optimizer_scheduler(args)
    density = GaussianDensityTorch()  # Used for anomaly score calculation
    net.to(args.device)

    # Get specific model implementation based on method argument
    model = get_model(args, net, optimizer, scheduler)

    # Initialize storage for tracking tasks and distributions
    dataloaders_train = []  # Stores training dataloaders for all tasks
    dataloaders_test = []  # Stores test dataloaders for all tasks
    learned_tasks = []  # Keeps track of completed tasks
    all_test_filenames = []  # Stores filenames for testing

    # Statistics storage for calculating final distribution
    task_wise_mean = []  # Stores mean embeddings per task
    task_wise_cov = []  # Stores covariance matrices per task
    task_wise_train_data_nums = []  # Stores number of samples per task

    # Main training loop over tasks
    for t in range(args.dataset.n_tasks):
        print('---' * 10, f'Task:{t}', '---' * 10)

        # Get dataloaders for current task and update storage
        # First, passes empty lists
        train_dataloader, dataloaders_train, dataloaders_test, learned_tasks, data_train_nums, all_test_filenames, train_data, test_data = \
            get_dataloaders(args, t, dataloaders_train, dataloaders_test, learned_tasks, all_test_filenames)
        task_wise_train_data_nums.append(data_train_nums)

        # Training loop for current task
        net.train()
        for epoch in tqdm(range(args.train.num_epochs)):
            one_epoch_embeds = []  # Stores embeddings from current epoch

            for batch_idx, (data) in enumerate(train_dataloader):
                inputs, labels = get_inputs_labels(data)
                print(labels)

            model(epoch, inputs, labels, one_epoch_embeds, t, extra_para=None)

        # Periodic evaluation during training
        if args.train.test_epochs > 0 and (epoch + 1) % args.train.test_epochs == 0:
            net.eval()
            # Update density estimation with current embeddings
            density = model.training_epoch(
                density,
                one_epoch_embeds,
                task_wise_mean,
                task_wise_cov,
                task_wise_train_data_nums,
                t
            )
            # Evaluate model on all learned tasks
            eval_model(args, epoch, dataloaders_test, learned_tasks, net, density)
            net.train()

    # Save final model and density estimator
    if args.save_checkpoint:
        torch.save(net, f'{args.save_path}/net.pth')
        torch.save(density, f'{args.save_path}/density.pth')


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    with warnings.catch_warnings(action="ignore"):
        args = get_args()
        print(type(args))
        print(type(args.device))
        print(type(args.dataset.n_tasks))
        # main(args)
