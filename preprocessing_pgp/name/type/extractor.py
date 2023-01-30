"""
Module to extract type from name
"""

from time import time

import pandas as pd
from flashtext import KeywordProcessor
from halo import Halo

from preprocessing_pgp.name.preprocess import preprocess_df
from preprocessing_pgp.name.accent_typing_formatter import remove_accent_typing
from preprocessing_pgp.utils import (
    parallelize_dataframe,
    sep_display
)
from preprocessing_pgp.name.type.const import (
    NAME_TYPE_DATA
)


class TypeExtractor:
    """
    Class contains function to support for type extraction from name
    """

    def __init__(self) -> None:
        self.available_levels = NAME_TYPE_DATA.keys()
        self.__generate_type_kws()

    def __generate_type_kws(self):
        if hasattr(self, 'type_kws'):
            return
        self.type_kws = {}
        for level in self.available_levels:
            level_kws = KeywordProcessor(case_sensitive=True)
            level_types = self.__get_available_types_by_level(level)

            level_kws.add_keywords_from_dict(
                dict(zip(
                    level_types,
                    [self.__get_unique_terms_in_ctype(ctype, level)
                     for ctype in level_types]
                ))
            )
            self.type_kws[level] = level_kws

    def __get_available_types_by_level(
        self,
        level: str = 'lv1'
    ):
        return NAME_TYPE_DATA[level]['ctype'].unique().tolist()

    def __get_unique_terms_in_ctype(
        self,
        ctype: str = 'company',
        level: str = 'lv1',
    ):
        return NAME_TYPE_DATA[level]\
            .query(f'ctype == "{ctype}"')['term']\
            .unique().tolist()

    def extract_type(
        self,
        name: str,
        level: str = 'lv1'
    ) -> str:
        """
        Extract name type from input name

        Parameters
        ----------
        name : str
            The input name to extract type from

        level : str
            The level to search for keyword pattern from,
            by default 'lv1' -- Currently there are 2 level 'lv1' and 'lv2'

        Returns
        -------
        str
            The first type extract from the name
        """
        if not hasattr(self, 'type_kws'):
            self.__generate_type_kws()

        results = self.type_kws[level].extract_keywords(name)

        if not results:
            return 'customer'
        return results[0]


@Halo(
    text='Formatting names',
    color='cyan',
    spinner='dots7',
    text_color='magenta'
)
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


@Halo(
    text='Extracting customer type',
    color='cyan',
    spinner='dots7',
    text_color='magenta'
)
def extract_ctype(
    data: pd.DataFrame,
    name_col: str = 'de_name',
    level:str = 'lv1'
) -> pd.DataFrame:
    """
    Perform name-type extraction from formatted name col

    Parameters
    ----------
    data : pd.DataFrame
        The original dataframe contains the formatted name col
    name_col : str, optional
        The formatted name col, by default 'de_name'
    level : str, optional
        The level to process type extraction, by default 'lv1'

    Returns
    -------
    pd.DataFrame
        The new data contains additional columns:

        * `customer_type` contains the type of customer extracted from name
    """
    type_extractor = TypeExtractor()
    extracted_data = data.copy()
    extracted_data['customer_type'] = extracted_data[name_col]\
        .apply(lambda name:
            type_extractor.extract_type(name, level)
        )

    return extracted_data


def process_extract_type(
    data: pd.DataFrame,
    name_col: str = 'name',
    level: str = 'lv1',
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
    level : str, optional
        The level to process type extraction, by default 'lv1'
    n_cores : int
        The number of cores used to run parallel, by default 1 core will be used

    Returns
    -------
    pd.DataFrame
        Final data contains additional columns:

        * `customer_type` contains type of customer extracted from `name` column
    """
    na_data = data[data[name_col].isna()].copy(deep=True)
    cleaned_data = data[data[name_col].notna()].copy(deep=True)

    # ? Format name
    start_time = time()
    if n_cores == 1:
        formatted_data = format_names(
            cleaned_data,
            name_col=name_col
        )
    else:
        formatted_data = parallelize_dataframe(
            cleaned_data,
            format_names,
            n_cores=n_cores,
            name_col=name_col
        )
    format_time = time() - start_time
    print(
        f"Formatting names takes {int(format_time)//60}m{int(format_time)%60}s")
    sep_display()

    # ? Extract name type
    start_time = time()
    if n_cores == 1:
        extracted_data = extract_ctype(
            formatted_data,
            name_col=f'de_{name_col}',
            level=level
        )
    else:
        extracted_data = parallelize_dataframe(
            formatted_data,
            extract_ctype,
            n_cores=n_cores,
            level=level,
            name_col=f'de_{name_col}',
        )
    extract_time = time() - start_time
    print(
        f"Extracting customer's type takes {int(extract_time)//60}m{int(extract_time)%60}s")
    sep_display()

    # ? Drop clean_name column
    extracted_data = extracted_data.drop(columns=[f'de_{name_col}'])

    # ? Combined with Na data
    final_data = pd.concat([extracted_data, na_data])

    return final_data
