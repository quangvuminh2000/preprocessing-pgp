"""
Module to extract type from name
"""

from time import time
from configparser import ConfigParser #! For reading .ini file

import pandas as pd

from preprocessing_pgp.name.preprocess import preprocess_df
from preprocessing_pgp.name.accent_typing_formatter import remove_accent_typing
from preprocessing_pgp.utils import (
    parallelize_dataframe,
    sep_display
)
from preprocessing_pgp.name.type.const import (
    NAME_TYPE_CONFIG,
    EXCLUDE_REGEX,
    TYPE_NAMES,
    KEYWORDS
)


class TypeExtractor:
    """
    Class contains function to support for type extraction from name
    """
    def __init__(self) -> None:
        self.config = NAME_TYPE_CONFIG
        self.exclude_dict = EXCLUDE_REGEX
        self.type_name_dict = TYPE_NAMES
        self.kws_mapper = KEYWORDS

    def extract_type(
        self,
        name: str
    ) -> str:
        """
        Extract name type from input name

        Parameters
        ----------
        name : str
            The input name to extract type from

        Returns
        -------
        str
            The type of the name
        """
        return ''


def format_names(
    data: pd.DataFrame,
    name_col: str = 'name'
) -> pd.DataFrame:
    """
    Format name with clean and encoded without accent

    Parameters
    ----------
    data : pd.DataFrame
        The input data contains the name records
    name_col : str, optional
        The column name in data holds the name records, by default 'name'

    Returns
    -------
    pd.DataFrame
        Final data contains additional columns:

        * `de_<name_col>` contains decoded & lowered name derived from `name`
    """
    clean_data = preprocess_df(data, name_col)
    clean_data[f'de_{name_col}'] = clean_data[name_col].apply(
        remove_accent_typing)
    clean_data[f'de_{name_col}'] = clean_data[f'de_{name_col}'].str.lower()

    return clean_data


def process_extract_type(
    data: pd.DataFrame,
    name_col: str = 'name',
    n_cores: int = 1
) -> pd.DataFrame:
    """
    Extract types from name records inputted from data

    Parameters
    ----------
    data : pd.DataFrame
        The input data contains the name records
    name_col : str, optional
        The column name in data holds the name records, by default 'name'

    Returns
    -------
    pd.DataFrame
        Final data contains additional columns:

        * `customer_type` contains type of customer extracted from `name` column
    """
    final_data = data.copy(deep=True)

    # ? Format name
    start_time = time()
    final_data = parallelize_dataframe(
        final_data,
        format_names,
        n_cores=n_cores,
        name_col=name_col
    )
    format_time = time() - start_time
    print(
        f"Formatting names takes {int(format_time)//60}m{int(format_time)%60}s")
    sep_display()

    # ? Extract name type
