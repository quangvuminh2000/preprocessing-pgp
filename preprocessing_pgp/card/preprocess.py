"""This module meant for preprocessing the card id before validation"""

import re
from typing import Tuple
from dataclasses import dataclass

import pandas as pd

from preprocessing_pgp.utils import sep_display


@dataclass
class CardIDCleaner:
    """
    Class contains cleansing functions for preprocessing card_id
    """

    def __remove_spaces(self, card_id: str) -> str:
        """
        Function to remove all spaces in card_id

        Parameters
        ----------
        card_id: str
            Input card_id to clean

        Returns
        -------
        str
            The output card_id without any spacing
        """

        # Remove spaces in between
        clean_card_id = re.sub(' +', '', card_id)
        clean_card_id = clean_card_id.strip()

        return clean_card_id

    def __remove_special_characters(self, card_id: str) -> str:
        """
        Removing special characters in card_id

        Parameters
        ----------
        card_id: str
            Input card_id to clean

        Returns
        -------
        str
            The output card_id without any punctuation
        """

        # translator = str.maketrans('', '', punctuation)

        # clean_card_id = card_id.translate(translator)

        clean_card_id = re.sub(r'\W+', '', card_id)

        return clean_card_id

    def clean_card(self, card_id: str) -> str:
        """
        Clean the card id with all the necessary cleaning functions

        Parameters
        ----------
        card_id: str
            Input card_id to clean

        Returns
        -------
        str
            The clean card without any special characters or spaces
        """
        clean_card_id = card_id.lower()
        clean_card_id = self.__remove_spaces(clean_card_id)
        clean_card_id = self.__remove_special_characters(clean_card_id)

        return clean_card_id


def extract_null_values(
    data: pd.DataFrame,
    by_col: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extracting NULL values from specific DataFrame

    Parameters
    ----------
    data : pd.DataFrame
        Basic DataFrame
    by_col : str
        Column to separate Null values

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        Tuple of Non-Null DataFrame and Null DataFrame
    """

    null_data = data[data[by_col].isna()].copy()

    non_null_data = data.dropna(subset=[by_col]).copy()

    return non_null_data, null_data


def clean_card_data(
    data: pd.DataFrame,
    card_col: str
) -> pd.DataFrame:
    """
    Preprocess card_id to clean format

    Parameters
    ----------
    data : pd.DataFrame
        Basic DataFrame
    card_col : str
        Column contains card_id to clean

    Returns
    -------
    pd.DataFrame
        Original DataFrame with a new columns named clean_<card_col>
    """

    card_cleaner = CardIDCleaner()

    clean_data = data.copy()

    print("Process cleaning card id...")
    clean_data[f'clean_{card_col}'] =\
        clean_data[card_col].apply(card_cleaner.clean_card)
    sep_display()

    return clean_data
