"""
Main function to run on the GPU Cluster
"""

from utils.train import train_model
from utils.eval import eval_model
import torch

def main():
    # Hyperparameters for training/testing
    torch.manual_seed(42)
    # Whether to do training, with which models, and on which datasets
    TRAIN = True
    EVAL = False
    models = {
        "DNE":True,
        "IUF":True,
        "CAD":False
    }
    datasets = {
        "MVTEC":True,
        "MTD":True
    }
    NUM_EPOCHS = 25
    BATCH_SIZE = 24
    LEARNING_RATE = 0.00025

    data_aug = {
        "color": [
            [0.20, 0.26],
            [0.26, 0.32],
            [0.32, 0.38],
            [0.38, 0.44],
            [0.44, 0.50],
            [0.50, 0.56],
            [0.56, 0.62],
            [0.62, 0.68],
            [0.68, 0.74],
            [0.74, 0.80]
        ],
        "blur": [
            [1, 0.5],
            [3, 1],
            [5, 1.5],
            [7, 2],
            [9, 2.5],
            [11, 3],
            [13, 3.5],
            [15, 4],
            [17, 4.5],
            [19, 5],
        ],
        "geometric": [
            [4, 2, 0.02, 4],
            [8, 4, 0.04, 8],
            [12, 6, 0.06, 12],
            [16, 8, 0.08, 16],
            [20, 10, 0.10, 20],
            [24, 12, 0.12, 24],
            [28, 14, 0.14, 28],
            [32, 16, 0.16, 32],
            [36, 18, 0.18, 36],
            [40, 20, 0.2, 40]
        ]
    }
    # Running Training Experiments
    if TRAIN:
        for model in models.keys():
            if models[model]:
                if datasets['MVTEC']:
                    train_model(model_type=model,
                                dataset='MVTEC',
                                num_epochs=NUM_EPOCHS,
                                batch_size=BATCH_SIZE,
                                criterion=torch.nn.CrossEntropyLoss(),
                                learning_rate=LEARNING_RATE
                                )
                elif datasets['MTD']:
                    for distortion in data_aug.keys():
                        train_model(model_type=model,
                                    dataset='MTD',
                                    num_epochs=NUM_EPOCHS,
                                    batch_size=BATCH_SIZE,
                                    criterion=torch.nn.CrossEntropyLoss(),
                                    learning_rate=LEARNING_RATE,
                                    tasks=data_aug[distortion],
                                    data_aug=distortion
                                    )

    # Running Evaluation Experiments
    if EVAL:
        for model in models.keys():
            if models[model]:
                task_data = eval_model(model_type=model,
                                       batch_size=BATCH_SIZE,
                                       data_aug=data_aug
                                       )

    return

if __name__ == "__main__":
    main()