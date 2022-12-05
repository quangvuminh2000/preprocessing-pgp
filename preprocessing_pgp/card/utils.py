"""Module provides utils functions for card validation"""

import re
import multiprocessing as mp
from typing import Tuple, Callable, List, Union
from string import ascii_lowercase
from string import punctuation

import pandas as pd
import numpy as np
from tqdm import tqdm

from preprocessing_pgp.card.const import (
    # Constant
    N_PROCESSES,
    # Personal ID
    OLD_PID_CODE_LENGTH,
    NEW_PID_CODE_LENGTH,
    POSSIBLE_GENDER_NUM,
    OLD_PID_REGION_CODE_NUMS,
    NEW_PID_REGION_CODE_NUMS,
    # Driver License
    DRIVER_LICENSE_LENGTH,
    DRIVER_LICENSE_ID_REGION_CODES,
    INVALID_DRIVER_LICENSE_PASSING_YEAR,
    INVALID_DRIVER_LICENSE_FIRST_YEAR_CHAR,
    VALID_DRIVER_LICENSE_LAST_YEAR_CHAR
)


def apply_multi_process(
    func: Callable,
    series: Union[pd.Series, str, np.ndarray]
) -> List:
    """
    Process multi-processing on every items of series with provided func

    Parameters
    ----------
    func : Callable
        Function to traverse through series, must have 1 input and 1 output
    series : Optional[pd.Series]
        Any series | np.Array() | list

    Returns
    -------
    List
        List of elements returned after apply the function
    """

    if isinstance(series, list):
        total_elem = len(series)
    else:
        total_elem = series.shape[0]

    with mp.Pool(N_PROCESSES) as pool:
        output = tqdm(
            pool.imap(func, series),
            total=total_elem
        )

    return output


def apply_progress_bar(
    func: Callable,
    series: pd.Series
) -> List:
    """
    Process apply with progress bar on every items of series with provided func

    Parameters
    ----------
    func : Callable
        Function to traverse through series, must have 1 input and 1 output
    series : pd.Series
        Any series of type pandas Series

    Returns
    -------
    List
        List of elements returned after apply the function
    """

    tqdm.pandas()

    return series.progress_apply(func)


def is_checker_valid(*checkers) -> bool:
    """
    Check if any of the checker is valid

    Returns
    -------
    bool
        Whether any of the checker is valid
    """

    return any(checkers)


def remove_spaces(sentence: str) -> str:
    """
    Function to remove all spaces in sentence

    Parameters
    ----------
    sentence : str
        The input sentence

    Returns
    -------
    str
        The output sentence without any spacing
    """

    # Remove spaces in between
    sentence = re.sub(' +', '', sentence)
    sentence = sentence.strip()

    return sentence


def remove_special_characters(sentence: str) -> str:
    """
    Removing special characters in string

    Parameters
    ----------
    sentence : str
        The sentence to remove punctuation

    Returns
    -------
    str
        The clean sentence without any punctuation
    """

    translator = str.maketrans('', '', punctuation)

    return sentence.translate(translator)


def check_contain_all_digit(
    card_id: str
) -> bool:
    """
    Simple function to check if the card_id contains all decimal

    Parameters
    ----------
    card_id : str
        The input card id

    Returns
    -------
    bool
        Whether the card id contains all decimal number
    """
    return card_id.isdecimal()


def check_non_digit(
    card_df: pd.DataFrame,
    card_col: str
) -> pd.Series:
    """
    Check if card contains any non_digit or not

    Parameters
    ----------
    card_df : pd.DataFrame
        The input card id DF
    card_col : str
        The column containing card id

    Returns
    -------
    pd.Series
        Series to verify id card
    """

    regex_non_digit = "|".join(list(ascii_lowercase))

    non_clean_mask = card_df[card_col].str.contains(regex_non_digit)

    return non_clean_mask


def check_card_length(
    card_df: pd.DataFrame, card_col: str = "card_id", valid_col: str = "is_valid"
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Validate card by length

    Parameters
    ----------
    card_df : pd.DataFrame
        The original DF containing card id column
    card_col : str, optional
        The column name in the DF that contains the id, by default 'card_id'

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        Correct card's length DF, Possibly correct card's length DF, and invalid card's length DF
    """

    # * Calculate card length
    card_df["card_length"] = card_df[card_col].str.len()

    # * Correct card_id length
    correct_length_mask = (
        card_df["card_length"].isin([OLD_PID_CODE_LENGTH, NEW_PID_CODE_LENGTH])
        & card_df["is_valid"].isna()
    )

    correct_card_df = card_df[correct_length_mask].copy()

    # * Possibly correct card_id length
    possible_length_mask = (
        card_df["card_length"].isin(
            [OLD_PID_CODE_LENGTH - 1, NEW_PID_CODE_LENGTH - 1])
        & card_df["is_valid"].isna()
    )

    possible_card_df = card_df[possible_length_mask].copy()

    possible_card_df[f"clean_{card_col}"] = "0" + possible_card_df[card_col]

    # * Invalid card length
    invalid_length_mask = ~(correct_length_mask | possible_length_mask)

    invalid_card_df = card_df[invalid_length_mask].copy()
    invalid_card_df[valid_col] = False

    return (correct_card_df, possible_card_df, invalid_card_df)


def is_old_card(card_id: str) -> bool:
    return len(card_id) == OLD_PID_CODE_LENGTH


def is_new_card(card_id: str) -> bool:
    return len(card_id) == NEW_PID_CODE_LENGTH


def is_valid_gender(gender_code: str) -> bool:
    return gender_code in POSSIBLE_GENDER_NUM


def is_valid_old_card(card_id: str) -> bool:
    if card_id[:2] in OLD_PID_REGION_CODE_NUMS:
        return True
        # gender_code = card_id[2]
        # return is_valid_gender(gender_code)

    if card_id[:3] in OLD_PID_REGION_CODE_NUMS:
        return True
        # gender_code = card_id[3]
        # return is_valid_gender(gender_code)

    return False


def is_valid_new_card(card_id: str) -> bool:
    if card_id[:3] in NEW_PID_REGION_CODE_NUMS:
        gender_code = card_id[3]
        return is_valid_gender(gender_code)

    return False


def is_valid_card(card_id: str) -> bool:
    """
    Check if the card is valid by syntax or not

    Parameters
    ----------
    card_id : str
        The input card id

    Returns
    -------
    bool
        Whether the card is valid by syntax
    """
    # old card
    if is_old_card(card_id):
        return is_valid_old_card(card_id)

    # new card
    if is_new_card(card_id):
        return is_valid_new_card(card_id)

    return False


# TODO Continue to create code for validate driver license

def is_valid_driver_license_length(card_id: str) -> bool:
    return len(card_id) == DRIVER_LICENSE_LENGTH


def is_valid_driver_license_region_code(region_code: str) -> bool:
    return region_code in DRIVER_LICENSE_ID_REGION_CODES


def is_valid_license_passing_year(passing_year: str) -> bool:
    return passing_year not in INVALID_DRIVER_LICENSE_PASSING_YEAR


def is_valid_driver_license(card_id: str) -> bool:
    """
    Check if the id is valid driver license id

    Parameters
    ----------
    card_id : str
        The input card id

    Returns
    -------
    bool
        Whether the card id is the valid driver license
    """

    if not is_valid_driver_license_length(card_id):
        return False

    region_code = card_id[:2]
    gender_code = card_id[2]
    passing_year = card_id[3:5]

    return (
        check_contain_all_digit(card_id) and
        is_valid_driver_license_region_code(region_code) and
        is_valid_gender(gender_code) and
        is_valid_license_passing_year(passing_year)
    )


def is_real_driver_license(card_id: str) -> bool:
    if not is_valid_driver_license_length(card_id):
        return False

    first_year_char = card_id[3]
    last_year_char = card_id[4]

    return (first_year_char not in INVALID_DRIVER_LICENSE_FIRST_YEAR_CHAR) and (last_year_char in VALID_DRIVER_LICENSE_LAST_YEAR_CHAR)
