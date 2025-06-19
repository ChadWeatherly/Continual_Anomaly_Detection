import time
import torch
from torch.utils.data import DataLoader
import plotly.graph_objects as go
from IPython.display import clear_output
from Methods.DNE.dne import DNE_Model
from Methods.IUF.iuf_model import IUF_Model, IUF_Loss
import datasets

def train_model(model_type: str,
                dataset: str,
                num_epochs: int,
                batch_size: int,
                **kwargs):
    """
    Train a model with given criterion and optimizer. Assumes it's running from the root folder,
    and saving of plots and models assumes this current directory structure.
    Args:
        model_type: a string of which model to train ("DNE", "IUF", "CAD")
        dataset: a string of which dataset to use ("MVTEC", "MTD")
        num_epochs: an int of how many epochs to train the model
        batch_size: an int of how many samples per batch to train the model
        kwargs: additional arguments. Current assumed ones are:
            - criterion
            - learning_rate
            - tasks, if using MTD, which should contain a list of the dataset augmentation windows
                    (will be passed to datasets.mvtec() as data_aug_params)
            - data_aug, if using MTD, which should contain a string of the type of data augmentation for MTD,
                        which should be a string in ["color", "blur", "geometric"

    Returns:
    """
    # Check input assertions
    assert model_type in ["DNE", "IUF", "CAD"]
    assert dataset in ["MVTEC", "MTD"]

    # kwargs takes any other parameters not explicitly stated above, and turns them
    # into a dict(). kwargs.get() allows us to search that dict, or if it doesn't exist,
    # provide a default option
    criterion = kwargs.get('criterion', torch.nn.CrossEntropyLoss())

    # Set up tasks
    if dataset == "MVTEC":
        tasks = ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut',
                       'leather', 'metal_nut', 'pill', 'screw', 'tile', 'toothbrush',
                       'transistor', 'wood', 'zipper']
    elif dataset == "MTD":
        tasks = kwargs.get('tasks')
        data_aug = kwargs.get('data_aug')
        assert data_aug in ["color", "blur", "geometric"]

    # Tracks losses across each experiment, then task.
    # Contains keys of: train_task_losses[unsupervised][task][epoch]
    train_task_losses = {}

    # Iterate through each unsupervised/supervised experiment
    for unsupervised in [True, False]:
        # A new model for each unsupervised/supervised task
        match model_type:
            case "DNE":
                model = DNE_Model()
            case "IUF":
                model = IUF_Model()
            case "CAD":
                pass
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=kwargs.get('learning_rate', 0.0001))

        # Keeps track of current experiment losses (unsupervised/supervised)
        exp_losses = {}
        # Iterate through tasks
        for t in range(len(tasks)):
            task = tasks[t]
            # Get data
            if dataset == "MVTEC":
                task_dataset = datasets.mvtec(train=True,
                                              unsupervised=unsupervised,
                                              task=task,
                                              transform=None)
                task_name = task
            elif dataset == "MTD":
                task_dataset = datasets.mtd(train=True,
                                            unsupervised=unsupervised,
                                            transform=None,
                                            data_aug=data_aug,
                                            data_aug_params=task)
                task_name = f"{data_aug}_"
                for param in task:
                    task_name += str(param).replace(".", "")
                    if param != task[-1]:
                        task_name += "_"

            dataloader = DataLoader(task_dataset,
                                    batch_size=batch_size,
                                    shuffle=True,
                                    collate_fn=datasets.collate)
            task_loss = []
            # Run through each epoch
            for e in range(num_epochs):
                start_time = time.time()
                clear_output(wait=False)
                print(f"Running {'unsupervised' if unsupervised else 'supervised'} experiment on {dataset} ({task_name}):")
                print(f"Epoch {e+1}/{num_epochs}")
                if e>2:
                    print("----------------------")
                    print('Previous Epoch Losses: ')
                    for i in [-3, -2, -1]:
                        print(f"{task_loss[i]}")
                    print("----------------------")
                    print("Last epoch time:")
                    mins = int(curr_epoch_time / 60)
                    secs = int(curr_epoch_time % 60)
                    print(f"{mins} minutes, {secs} seconds")

                loss = model.train_one_epoch(dataloader=dataloader,
                                             optimizer=optimizer,
                                             criterion=criterion,
                                             task_num=(t+1),
                                             update_z_epoch=True if (e+1)==num_epochs else False)
                task_loss.append(loss)
                curr_epoch_time = time.time() - start_time

            # Update current experiment loss with current task_loss list
            exp_losses[task_name] = task_loss

            if model_type == "DNE":
                # Save memory distribution for this task in self.memory
                model.update_memory()

            # After last epoch, save model params for this task (saves weights and memory, if DNE)
            model.save(f"./models/{model_type}/{dataset}/{"unsupervised" if unsupervised else "supervised"}/{task_name}_weights.pth")

        exp = "unsupervised" if unsupervised else "supervised"
        train_task_losses[exp] = exp_losses

    # Get training plots and save
    # Plot figure comparing experiments, where each experiment has its own plot, showing how the model trained on each task
    for exp in ['supervised', 'unsupervised']:
        fig = go.Figure()
        for task in tasks:
            if dataset == "MVTEC":
                task_name = task
            elif dataset == "MTD":
                task_name = f"{data_aug}_"
                for param in task:
                    task_name += str(param).replace(".", "")
                    if param != task[-1]:
                        task_name += "_"

            fig.add_trace(go.Scatter(y=train_task_losses[exp][task_name],
                                     mode='lines', name=task_name))
        fig.update_layout(title=f"Epoch Training Loss for {exp} experiment, per task",
                          xaxis_title="Epoch",
                          yaxis_title="Total Loss")
        fig.show()

    return model