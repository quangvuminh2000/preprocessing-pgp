"""
Module for filtering dataframe
"""

import pandas as pd


def get_difference_data(
    *datasets: pd.DataFrame
) -> pd.DataFrame:
    """
    Function to get differences data between datasets

    Returns
    -------
    pd.DataFrame
        The datasets to get the differences
    """
    difference_data = pd.concat(
        datasets,
        ignore_index=True
    ).drop_duplicates(keep=False)

    return difference_data
