"""
Module for updating dataframe
"""
from datetime import datetime, timedelta

import pandas as pd

from const import (
    hdfs,
    UNIFY_PATH,
)


def update_profile(
    cttv_name: str,
    day: str,
    new_profile: pd.DataFrame,
    unify_func,
    **kwargs
) -> pd.DataFrame:
    """
    Update and return the unify profile for current day
    """
    yesterday_str =\
        (datetime.strptime(day, '%Y-%m-%d')
         - timedelta(days=1)).strftime('%Y-%m-%d')
    profile_unify = pd.read_parquet(
        f'{UNIFY_PATH}/{cttv_name}.parquet/d={yesterday_str}',
        filesystem=hdfs
    )
    if new_profile.empty:
        return profile_unify

    new_profile_unify = unify_func(new_profile, **kwargs)

    profile_unify = pd.concat(
        [new_profile_unify, profile_unify],
        ignore_index=True
    )

    return profile_unify
