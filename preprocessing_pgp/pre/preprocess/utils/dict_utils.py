"""
Module for dictionary utilities
"""

import sys
from typing import List

import pandas as pd

sys.path.append('/bigdata/fdp/cdp/source/core_profile/preprocess/utils')
from const import (
    hdfs
)


def load_hdfs_data(
    path: str,
    date: str,
    cols: List[str] = []
) -> pd.DataFrame:
    if len(cols) > 0:
        return pd.read_parquet(
            path,
            columns=cols,
            filesystem=hdfs
        )

    return pd.read_parquet(
        path,
        filesystem=hdfs
    )
