"""
Module contains utility functions to help the email's functions
"""

import re
from typing import List, Tuple

import pandas as pd
from unidecode import unidecode


def split_email(email: str) -> List[str]:
    """
    Split email into email's name & group (default by `@`)

    Parameters
    ----------
    email : str
        The original email

    Returns
    -------
    List[str]
        The list contains email's `name` and `group`
    """
    if not email:
        return [None, None]

    split_result = email.split('@', maxsplit=1)

    if len(split_result) == 2:
        return split_result

    return [*split_result, None]

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

def clean_email_name(
    data:pd.DataFrame,
    name_col: str = 'email_name'
) -> pd.DataFrame:
    """
    Process cleaning email name data -- removing '.' and other processing

    Parameters
    ----------
    data : pd.DataFrame
        The original dataframe
    name_col : str, optional
        The column name contains email_name, by default 'email_name'

    Returns
    -------
    str
        Cleaned email name data with additional column `cleaned_email_name`
    """
    data[f'cleaned_{name_col}'] = ''

    # * Removing accented name
    data.loc[
        data[name_col].apply(is_name_accented),
        f'cleaned_{name_col}'
    ] = None

    # * Removing names with spaces
    data.loc[
        data[name_col].str.contains(r'\s+', regex=True),
        f'cleaned_{name_col}'
    ] = None

    # * Remove digit, '.' from name & lower name
    clean_data_mask = data[f'cleaned_{name_col}'].notna()
    data.loc[
        clean_data_mask,
        f'cleaned_{name_col}'
    ] = data.loc[
        clean_data_mask,
        name_col
    ].str.replace(r'\d+', '', regex=True)\
        .str.replace('.', '', regex=True)\
        .str.lower()

    # * Filter other case non-cleaned
    data.loc[
        data[f'cleaned_{name_col}'] == '',
        f'cleaned_{name_col}'
    ] = None

    return data

def sort_series_by_appearance(
    series: pd.Series
) -> pd.Series:
    """
    Sort series by the order of appearance in descending order

    Parameters
    ----------
    series : pd.Series
        The input series to sort for appearance

    Returns
    -------
    pd.Series
        The sorted descending series
    """

    name = series.name

    sorted_series = series.copy()\
        .value_counts().reset_index(name='count')\
        .rename(columns={'index': name})[name]

    return sorted_series

def extract_sub_string(
    sub_regex: str,
    string: str,
    take_first: bool = True
) -> Tuple[str, str]:
    """
    Extract substring from string given the regex

    Then return the sub_string and the remaining string

    If not found return empty string `''`

    Parameters
    ----------
    sub_regex : str
        The regex to extract from string
    string : str
        The input string to extract from
    take_first : bool
        Whether to take the first appearance, by default True

    Returns
    -------
    Tuple[str, str]
        `sub_string` and `remained string`
    """
    found_subs = re.findall(sub_regex, string)

    # * Not found any substring match
    if len(found_subs) == 0:
        return ('', string)

    # * Found at least one
    if take_first:
        sub_string = found_subs[0]
    else:
        sub_string = found_subs[-1]
    remain_string = re.sub(sub_string, '', string)

    return sub_string, remain_string
