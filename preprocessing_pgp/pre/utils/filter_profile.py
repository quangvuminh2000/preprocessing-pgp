"""
Module for filtering dataframe
"""

from typing import List, Tuple
from datetime import datetime, timedelta

import pandas as pd

from const import (
    RAW_PATH,
    UNIFY_PATH,
    hdfs
)


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


def load_profile(
    day: str,
    cttv_name: str,
    columns: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load old(yesterday) and new(today) profile for specific cttv

    Parameters
    ----------
    day : str
        The current day to update
    cttv_name : str
        The name of service group
    columns : List[str]
        List of columns to get from raw

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        The DFs of the current day and yesterday raw profile (today, yesterday)
    """
    yesterday_str =\
        (datetime.strptime(day, '%Y-%m-%d') - timedelta(days=1))\
        .strftime('%Y-%m-%d')

    today_profile = pd.read_parquet(
        f'{RAW_PATH}/{cttv_name}.parquet/d={day}',
        filesystem=hdfs,
        columns=columns
    )
    yesterday_profile = pd.read_parquet(
        f'{RAW_PATH}/{cttv_name}.parquet/d={yesterday_str}',
        filesystem=hdfs,
        columns=columns
    )

    return today_profile, yesterday_profile


def update_profile(
    day: str,
    cttv_name: str,
    difference_profile: pd.DataFrame,
    unify_func,
    **kwargs
) -> pd.DataFrame:
    """
    Update profile with specific unify function for specific cttv

    Parameters
    ----------
    day : str
        The current day to update
    cttv_name : str
        The name of service group
    difference_profile : pd.DataFrame
        The DF of difference profile
    unify_func : func
        The function to unify profile in CTTV

    Returns
    -------
    pd.DataFrame
        The unify profile for the current day
    """
    yesterday_str =\
        (datetime.strptime(day, '%Y-%m-%d') - timedelta(days=1))\
        .strftime('%Y-%m-%d')
    yesterday_unify_profile = pd.read_parquet(
        f'{UNIFY_PATH}/{cttv_name}.parquet/d={yesterday_str}'
    )

    if difference_profile.empty:
        return yesterday_unify_profile

    update_unify_profile = unify_func(difference_profile, **kwargs)
    today_unify_profile = pd.concat([
        update_unify_profile,
        yesterday_unify_profile
    ], ignore_index=True)

    return today_unify_profile
