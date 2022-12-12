"""
File containing code that related to n-level extraction of address
"""

from abc import ABC
from dataclasses import dataclass
from typing import List, Tuple
from copy import deepcopy

from flashtext import KeywordProcessor

from preprocessing_pgp.address.scripts.const import (
    METHOD_REFER_DICT,
    LOCATION_ENRICH_DICT
)


@dataclass
class LevelMethod(ABC):
    """
    Class contains information of extract level and functions to process on each level

    * Level 1: City, Countryside
    * Level 2: District
    * Level 3: Ward
    """

    @staticmethod
    def get_level_keywords(level: int = 1) -> List:
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
        for method in METHOD_REFER_DICT[level]:
            method_kw =\
                LevelMethod.generate_keyword_processor(method)
            level_kws.append(method_kw)

        return level_kws

    @staticmethod
    def generate_keyword_processor(method: str) -> KeywordProcessor:
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

    @staticmethod
    def get_match_keyword(query: str, kw_processor: KeywordProcessor) -> str:
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
            Last match keyword found in query, if not found return empty string `''`
        """
        found_keywords = kw_processor.extract_keywords(query)

        if len(found_keywords) > 0:
            return found_keywords[-1]

        return ''

    @staticmethod
    def extract_all_levels(address: str) -> List:
        """
        _summary_

        Parameters
        ----------
        address : str
            _description_

        Returns
        -------
        List
            _description_
        """

        remained_address = deepcopy(address)

        for level, methods in METHOD_REFER_DICT.items():
            level_pattern, remained_address =\
                LevelMethod.extract_by_level(remained_address)

    @staticmethod
    def extract_by_level(address: str) -> Tuple[str, str]:
