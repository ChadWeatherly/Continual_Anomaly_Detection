import time
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import plotly.graph_objects as go
from IPython.display import clear_output
from Methods.DNE.dne import DNE_Model
from Methods.IUF.iuf import IUF_Model, IUF_Loss
import datasets

def eval_model(model_type: str,
                batch_size: int,
               metrics: list[str],
                **kwargs):
    """
    Test a model on all experiments. Assumes it's running from the root folder,
    and saving of plots and models assumes this current directory structure.
    Args:
        model_type: a string of which model to train ("DNE", "IUF", "UCAD")
        batch_size: an int of how many samples per batch to train the model
        metrics: a list of strings of which metrics to use for evaluating the model
        kwargs: additional arguments. Current assumed ones are:
            - data_aug, for MTD, which should contain a dict of the type of data augmentation for MTD,
                        with keys being any combination of ["color", "blur", "geometric"]
                        and each entry being a list of lists, where each individual list
                        is a set of data augmentation parameters

    Returns:
    """
    # kwargs takes any other parameters not explicitly stated above, and turns them
    # into a dict(). kwargs.get() allows us to search that dict, or if it doesn't exist,
    # provide a default option

    # Check input assertions
    assert model_type in ["DNE", "IUF", "UCAD"]

    # Iterate through each dataset
    for dataset in ["MTD", "MVTEC"]:

        # Go through each dataset
        if dataset == "MVTEC":
            tasks = ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut',
                     'leather', 'metal_nut', 'pill', 'screw', 'tile', 'toothbrush',
                     'transistor', 'wood', 'zipper']

            # Iterate through each unsupervised/supervised experiment
            for unsupervised in [True, False]:
                # Load in model
                # Get model
                match model_type:
                    case "DNE":
                        model = DNE_Model()
                    case "IUF":
                        model = IUF_Model()
                    case "UCAD":
                        pass
                model.load(f"./models/{model_type}/{model_type}_{dataset}_{"unsupervised" if unsupervised else "supervised"}_weights.pth")

                # If DNE, we need to generate our global distribution for inference
                if model_type == "DNE":
                    model.generate_global_dist()

                # Iterate through tasks
                for t in range(len(tasks)):

                    # Get task
                    task = tasks[t]

                    clear_output(wait=False)
                    # Print status values
                    print(f"Testing {model_type}-MVTEC-{'Unsupervised' if unsupervised else 'Supervised'}",
                          "--------------------",
                          f"Current Task: {task}",
                          sep="\n")

                    # Run through current task
                    test_dataset = datasets.mvtec(train=False, task=task, unsupervised=unsupervised)
                    test_dataloader = DataLoader(test_dataset, batch_size=batch_size,
                                                 shuffle=True, collate_fn=datasets.collate)
                    # Evaluate one epoch of the test_dataset
                    model_output = model.calc_results(test_dataloader,
                                                      "MVTEC",
                                                      task,
                                                      tasks,
                                                      "unsupervised" if unsupervised else "supervised",
                                                      metrics)
        elif dataset == "MTD":
            task_dict = kwargs.get('data_aug')
            for distortion in task_dict.keys():
                tasks = [] # list of tuples, containing tuples of (distortion, data_aug_params)
                task_names = []
                for task in task_dict[distortion]:
                    task_name = distortion + "_"
                    tasks.append((distortion, task))
                    for param in task:
                        task_name += (str(param).replace(".", ""))
                        if param != task[-1]:
                            task_name += "_"
                    task_names.append(task_name)

                # Iterate through each unsupervised/supervised experiment
                for unsupervised in [True, False]:
                    # Load in model
                    # Get model
                    match model_type:
                        case "DNE":
                            model = DNE_Model()
                        case "IUF":
                            model = IUF_Model()
                        case "UCAD":
                            pass
                    model.load(f"./models/{model_type}/{model_type}_{dataset}_{distortion}_{"unsupervised" if unsupervised else "supervised"}_weights.pth")

                    # If DNE, we need to generate our global distribution for inference
                    if model_type == "DNE":
                        model.generate_global_dist()

                    # Iterate through tasks
                    for t in range(len(tasks)):

                        # Get task
                        task = tasks[t]
                        # Get task name, useful for saving data
                        task_name = task_names[t]

                        clear_output(wait=False)
                        # Print status values
                        print(f"Testing {model_type}-MTD-{'Unsupervised' if unsupervised else 'Supervised'}",
                              "--------------------",
                              f"Current Task: {task}",
                              sep="\n")

                        # Run through current and all previous tasks
                        test_dataset = datasets.mtd(train=False, unsupervised=unsupervised,
                                                        data_aug=task[0], data_aug_params=task[1])
                        test_dataloader = DataLoader(test_dataset, batch_size=batch_size,
                                                     shuffle=True, collate_fn=datasets.collate)
                        # Evaluate one epoch of the test_dataset
                        model_output = model.calc_results(test_dataloader,
                                                          "MTD",
                                                          task_name,
                                                          task_names,
                                                          "unsupervised" if unsupervised else "supervised",
                                                          metrics)

                        break
                    break
                break
            break

    return model_output