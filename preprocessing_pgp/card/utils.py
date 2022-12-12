"""Module provides utils functions for card validation"""

import re
import multiprocessing as mp
from typing import Tuple, Callable, List, Union
from string import ascii_lowercase
from string import punctuation

import pandas as pd
import numpy as np
from tqdm import tqdm


def digit_to_year_string(digit: int) -> str:
    """
    Convert digit in range (0, 99) to 2 digits birth string

    Parameters
    ----------
    digit : int
        The digit representing dob, must be in range `0-99` else will be thrown error

    Returns
    -------
    str
        The string representing the birth year
    """

    if not 0 <= digit <= 99:
        raise AttributeError("DOB Digit must be in range 0 to 99!")

    birth_str = f'0{digit}' if digit < 10 else str(digit)

    return birth_str


def is_checker_valid(checkers: List) -> bool:
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
