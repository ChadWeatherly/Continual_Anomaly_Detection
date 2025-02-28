import os
import numpy as np
import random

def create_mvtec_supervised():
    """
    On average, we have about 17 images per anomaly type in each category.
    In practice, we don't often have access to that many anomalous cases,
    so let's use 20% for testing and 80% for training.
    """
    train_frac = 0.2
    path = "datasets/mvtec_anomaly_detection/supervised/"
    for cat in os.listdir(path): # Iterate through categories
        
        if "." not in cat:
            for anom_class in os.listdir(f"{path}{cat}/test"): # Iterate through category anomaly types
                if anom_class != "good":
                    for img_file in os.listdir(f"{path}{cat}/test/{anom_class}"): # Iterate through each image
                        if img_file.endswith(".png"):
                            if random.random() <= train_frac:
                                old_path = f"{path}{cat}/test/{anom_class}"
                                new_path = f"{path}{cat}/train/{anom_class}/"
                                if not os.path.isdir(new_path):
                                    print(new_path)
                                    os.makedirs(new_path, 0o777)
                                    print('made dir', sep="\n")
                                # os.rename(path+cat+"/test/"+anom_class+"/"+img_file, path+cat+"/train/"+anom_class+"/"+img_file)


    return