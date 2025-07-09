import pandas as pd

def create_df(dataset, all_tasks, metric):
    """
    Creates an empty dataframe for the models to populate with metrics and save
    Args:
        dataset (str): name of the dataset, either 'MVTEC' or 'MTD'
        all_tasks: (list[str]) list of all task names for column naming
        metric: (str): name of the metric

    Returns: empty dataframe
    """
    df = pd.DataFrame()
