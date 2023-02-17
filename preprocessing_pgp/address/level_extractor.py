"""
File containing code that related to n-level extraction of address
"""

from typing import List, Tuple, Dict
from copy import deepcopy

import pandas as pd
from flashtext import KeywordProcessor
from halo import Halo

from preprocessing_pgp.address.utils import (
    flatten_list,
    remove_substr,
    create_dependent_query
)
from preprocessing_pgp.address.const import (
    METHOD_REFER_DICT,
    LOCATION_ENRICH_DICT
)


class LevelExtractor:
    """
    Class contains information of extract level and functions to process on each level

    * Level 1: City, Countryside
    * Level 2: District
    * Level 3: Ward
    """

    def __init__(self) -> None:
        self.avail_levels = METHOD_REFER_DICT.keys()
        self.avail_methods = flatten_list(METHOD_REFER_DICT.values())
        self.keyword_refer_dict =\
            dict(zip(
                self.avail_methods,
                [self._generate_keyword_processor(method)
                 for method in self.avail_methods]
            ))

    def _get_level_methods(self, level) -> List:
        """
        _summary_

        Parameters
        ----------
        level : _type_
            _description_

        Returns
        -------
        List
            _description_
        """
        return METHOD_REFER_DICT[level]

    def _get_level_keywords(self, level: int = 1) -> List:
        """
        Function to get all keywords of each specific level

        Parameters
        ----------
        level : int, optional
            The number of level (1-3) to generate method keywords, by default 1

        Returns
        -------
            The output list of keywords for each of the level's methods
        """
        level_kws = []
        for method in self._get_level_methods(level):
            method_kw =\
                self._generate_keyword_processor(method)
            level_kws.append(method_kw)

        return level_kws

    def _generate_keyword_processor(self, method: str) -> KeywordProcessor:
        """
        Function to generate `KeywordProcessor` object
        to the specific `method`
        using the generated dictionary of keywords

        Parameters
        ----------
        method : str
            The method string refer to the dictionary column name

        Returns
        -------
        KeywordProcessor
            `KeywordProcessor` object that contains all the keywords of the method
        """
        unique_method_kws =\
            LOCATION_ENRICH_DICT[method].unique().tolist()

        keyword_processor = KeywordProcessor(case_sensitive=True)
        keyword_processor.add_keywords_from_list(unique_method_kws)

        return keyword_processor

    def _get_match_keyword(
        self,
        query: str,
        kw_processor: KeywordProcessor
    ) -> str:
        """
        Function to get the last keyword found in `query` string

        Parameters
        ----------
        query : str
            The input query string containing the address
        kw_processor : KeywordProcessor
            The keyword processor for specific method in dictionary (norm, abbrev, ...)

        Returns
        -------
        str
            Last match keyword found in query, if not found return None
        """
        found_keywords = kw_processor.extract_keywords(query)

        if len(found_keywords) > 0:
            return found_keywords[-1]

        return None

    def extract_all_levels(self, address: str) -> Tuple[Dict, str]:
        """
        Traverse through all possible level from lowest to highest
        to check and find `best pattern` in each level
        and `return` as a `dictionary of pattern` found at each level with `remaining address`

        Parameters
        ----------
        address : str
            The unified address that has been `lowered` and `unidecode`

        Returns
        -------
        Tuple[Dict, str]
            The output contains:
            * a dictionary of `pattern` found at each level
            * `remained address`
            * a dictionary of `best pattern` found at each level
        """

        remained_address = deepcopy(address)
        found_patterns =\
            dict(zip(
                self.avail_levels,
                [None]*len(self.avail_levels)
            ))
        best_patterns = deepcopy(found_patterns)

        dependents = []
        for level in self.avail_levels:
            (level_pattern,
             remained_address,
             level_method,
             level_best_pattern) =\
                self._extract_by_level(remained_address, level, *dependents)

            found_patterns[level] = level_pattern
            best_patterns[level] = level_best_pattern
            dependents.append((level_pattern, level_method))

        return found_patterns, remained_address, best_patterns

    def __is_query_exist(
        self,
        query: str
    ) -> bool:
        """
        Helper function to check whether the query is possibly found in location data
        """
        n_result = LOCATION_ENRICH_DICT.query(query).shape[0]

        return n_result > 0

    def __make_dependent_query(
        self,
        *dependents
    ) -> str:
        """
        Helper to make search query from dependents
        """
        query = ''
        # Making dependent queries
        if len(dependents) > 0:
            dependent_queries = []
            for d_term, d_method in dependents:
                if d_term is not None:
                    term_query = f'{d_method} == "{d_term}"'
                    dependent_queries.append(term_query)

            query = create_dependent_query(*dependent_queries)

        return query

    def __is_correct_dependent(
        self,
        *dependents
    ) -> bool:
        """
        Helper function to check whether the dependencies are all correct
        """
        query = self.__make_dependent_query(*dependents)

        if query != '':
            return self.__is_query_exist(query)

        return False

    def _extract_by_level(
        self,
        address: str,
        level: int,
        *dependents
    ) -> Tuple[str, str, str, str]:
        """
        Extract address with list of `method`
        at the specific `level`
        to find the best matched pattern

        Parameters
        ----------
        address : str
            The unified address that has been `lowered` and `unidecode`
        level : int
            The level which is currently traversed from

        Returns
        -------
        Tuple[str, str, str, str]
            The output contains:
            * `pattern` found within specific level
            * `remained address`
            * `method` which retrieved the pattern
            * `best pattern` unified by `pattern`
        """
        level_methods = self._get_level_methods(level)

        for method in level_methods:
            pattern_found, remained_address =\
                self._extract_by_method(address, method)

            # Found something then return
            if (pattern_found is not None)\
                    and (len(pattern_found) > 0):
                if self.__is_correct_dependent(*dependents, (pattern_found, method)):
                    best_pattern = self.__trace_match_pattern(
                        pattern_found,
                        method,
                        *dependents
                    )
                    return pattern_found, remained_address, method, best_pattern

        return None, address, None, None

    def _extract_by_method(
        self,
        address: str,
        method: str
    ) -> Tuple[str, str]:
        """
        Extract the address with `keywords` of the specific method

        Parameters
        ----------
        address : str
            The unified address that has been `lowered` and `unidecode`
        method : str
            The method of level which is currently being explored

        Returns
        -------
        Tuple[str, str]
            The output contains:
            * `pattern` found within specific method
            * `remained address`
        """
        method_kw = self.keyword_refer_dict[method]
        match_pattern = self._get_match_keyword(address, method_kw)
        remained_address = remove_substr(address, match_pattern)

        return match_pattern, remained_address

    def __trace_match_pattern(
        self,
        pattern: str,
        method: str,
        *dependents
    ) -> str:
        """
        Helper function to trace back best pattern found
        """
        query = self.__make_dependent_query(*dependents, (pattern, method))

        if not self.__is_query_exist(query):
            return None

        level_col = method[:3]

        best_match = self.__trace_best_match(query, level_col)

        return best_match

    def __trace_best_match(
        self,
        query: str,
        level_col: str
    ) -> str:
        """
        Helper to get the best match consist of:

        1. Exist
        2. Unique found
        """
        found_terms = LOCATION_ENRICH_DICT.query(query)[level_col].unique()

        n_found = found_terms.shape[0]
        if n_found == 0 or n_found > 1:
            return None

        return found_terms[0]


@Halo(
    text='Extracting address',
    color='cyan',
    spinner='dots7',
    text_color='magenta'
)
def extract_vi_address_by_level(
    data:  pd.DataFrame,
    address_col: str
) -> pd.DataFrame:
    """
    Function to extract vietnamese address into each specific level found by pattern

    Parameters
    ----------
    data : pd.DataFrame
        Raw data containing the address
    address_col : str
        The raw address column that need cleansing and unifying

    Returns
    -------
    pd.DataFrame
        Final data with new columns:

        * `level_1`: city, countryside found
        * `best_level_1`: beautified city, countryside found
        * `level_2`: district found
        * `best_level_2`: beautified district found
        * `level_3`: ward found
        * `best_level_3`: beautified ward found
        * `remained_address`: the remaining in the address
    """

    extractor = LevelExtractor()

    extracted_data = data.copy()

    extracted_results =\
        extracted_data[address_col].apply(
            extractor.extract_all_levels
        )

    for level in extractor.avail_levels:
        extracted_data[f'level_{level}'] =\
            [result[0][level]
             for result in extracted_results]

        extracted_data[f'best_level_{level}'] =\
            [result[-1][level]
             for result in extracted_results]

    extracted_data['remained_address'] =\
        [result[1]
         for result in extracted_results]

    return extracted_data
