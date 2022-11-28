from typing import Tuple
from string import ascii_lowercase
import re

import pandas as pd

from preprocessing_pgp.card.const import (
    OLD_CODE_LENGTH,
    NEW_CODE_LENGTH,
    POSSIBLE_GENDER_NUM,
    OLD_CODE_NUMS,
    NEW_CODE_NUMS,
)


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


def check_contain_digit(
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
        card_df["card_length"].isin([OLD_CODE_LENGTH, NEW_CODE_LENGTH])
        & card_df["is_valid"].isna()
    )

    correct_card_df = card_df[correct_length_mask].copy()

    # * Possibly correct card_id length
    possible_length_mask = (
        card_df["card_length"].isin([OLD_CODE_LENGTH - 1, NEW_CODE_LENGTH - 1])
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
    return len(card_id) == OLD_CODE_LENGTH


def is_new_card(card_id: str) -> bool:
    return len(card_id) == NEW_CODE_LENGTH


def is_valid_gender(gender_code: str) -> bool:
    return gender_code in POSSIBLE_GENDER_NUM


def is_valid_old_card(card_id: str) -> bool:
    if card_id[:2] in OLD_CODE_NUMS:
        gender_code = card_id[2]
        return is_valid_gender(gender_code)

    if card_id[:3] in OLD_CODE_NUMS:
        gender_code = card_id[3]
        return is_valid_gender(gender_code)

    return False


def is_valid_new_card(card_id: str) -> bool:
    if card_id[:3] in NEW_CODE_NUMS:
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
