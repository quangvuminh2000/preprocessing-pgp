"""
Module contains tools for processing name
"""

import pandas as pd
from unidecode import unidecode

from preprocessing_pgp.name.const import NICKNAME_REGEX


def remove_nicknames(
    name_df: pd.DataFrame,
    name_col: str
) -> pd.DataFrame:
    """
    Remove nicknames in name column given the data

    Parameters
    ----------
    name_df : pd.DataFrame
        The data process
    name_col : str
        The name column in data to remove nickname

    Returns
    -------
    pd.DataFrame
        Cleaned data without nickname -- added new column `clean_name_col`
    """
    name_df[name_col] =\
        name_df[name_col].str.replace(NICKNAME_REGEX, '', regex=True)\
        .str.strip()

    return name_df


def is_name_accented(name: str) -> bool:
    """
    Check whether the name is accented or not

    Parameters
    ----------
    name : str
        The input name

    Returns
    -------
    bool
        Whether the name is accented
    """
    return unidecode(name) != name
