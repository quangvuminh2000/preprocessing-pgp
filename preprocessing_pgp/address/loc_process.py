"""
File support extracting code for each location extracted in extractor
"""
from typing import List, Dict

import pandas as pd
from halo import Halo

from preprocessing_pgp.address.const import (
    LOCATION_CODE_DICT,
    LEVEL_VI_COLUMN_DICT,
    LEVEL_CODE_COLUMN_DICT,
    AVAIL_LEVELS
)
from preprocessing_pgp.address.utils import (
    create_dependent_query,
    is_empty_string
)


class LocationCode:
    def __init__(self) -> None:
        self.loc_code_dict = LOCATION_CODE_DICT.copy()
        self.avail_levels = LEVEL_VI_COLUMN_DICT.keys()

        self.__unify_dictionary()

    def __unify_dictionary(self):
        """
        Unifying location code dict to titled names
        """
        for level in self.avail_levels:
            level_col = LEVEL_VI_COLUMN_DICT[level]
            self.loc_code_dict[level_col] = self.loc_code_dict[level_col].str.title(
            )

    def __get_level_col(
        self,
        level: int
    ) -> str:
        """
        Helper to get back the level column
        """
        return LEVEL_VI_COLUMN_DICT[level]

    def __get_level_code_col(
        self,
        level: int
    ) -> str:
        """
        Helper to get back the level code column
        """
        return LEVEL_CODE_COLUMN_DICT[level]

    def get_level_code(
        self,
        components: Dict,
    ) -> Dict:
        """
        Function to get the `code` of all possible levels

        * The provided `components` must have
        same length as the number of available levels (currently 3)

        Parameters
        ----------
        components : Dict
            `Dictionary` containing:
            * `key`: level
            * `value`: best level name

        Returns
        -------
        Dict
            The codes dictionary for each level
        """

        # * Create query to trace for location id
        component_queries = []
        for level in self.avail_levels:
            level_col = self.__get_level_col(level)
            best_lvl_name = components[level]
            if best_lvl_name is not None:
                level_query = f'{level_col} == "{best_lvl_name.title()}"'
                component_queries.append(level_query)

        trace_loc_id_query = create_dependent_query(*component_queries)

        # * Trace back location id
        level_codes = dict(
            zip(self.avail_levels,
                [None]*len(self.avail_levels))
        )
        if is_empty_string(trace_loc_id_query):
            return level_codes

        matches = self.loc_code_dict.query(trace_loc_id_query)

        if matches.shape[0] == 0:
            return level_codes

        for level in self.avail_levels:
            level_col = self.__get_level_code_col(level)
            match_code = matches[level_col].values[0]
            level_codes[level] = match_code

        # * Order the trace from lower level to higher level
        for level in self.avail_levels:
            if components[level] is None:
                for inv_level in reversed(list(self.avail_levels)):
                    if inv_level == level:
                        break
                    level_codes[inv_level] = None

                level_codes[level] = None
                break

        return level_codes


@Halo(
    text='Generating location code',
    color='cyan',
    spinner='dots7',
    text_color='magenta'
)
def generate_loc_code(
    data: pd.DataFrame,
    best_lvl_cols: List[str]
) -> pd.DataFrame:
    """
    Function to retrieve location code for each level:
    * `level_1_code`: code for city
    * `level_2_code`: code for district
    * `level_3_code`: code for ward

    Parameters
    ----------
    data : pd.DataFrame
        Raw data containing the address
    best_lvl_cols : List[str]
        The best extracted level column to retrieve code

    Returns
    -------
    pd.DataFrame
        New dataframe with new columns representing level code
    """

    code_generator = LocationCode()

    generated_data = data.copy()

    row_codes = generated_data.apply(
        lambda row: code_generator.get_level_code(
            dict(zip(
                AVAIL_LEVELS,
                [row[col] for col in best_lvl_cols]
            ))
        ),
        axis='columns'
    )

    for level in AVAIL_LEVELS:
        generated_data[f'level_{level}_code'] = [
            code[level] for code in row_codes]

    return generated_data
