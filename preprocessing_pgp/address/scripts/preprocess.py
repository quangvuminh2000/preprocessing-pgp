"""
This module meant for preprocessing (cleansing and unifying) the address before putting into extraction
"""

import re
from typing import Dict
from copy import deepcopy

import pandas as pd
from tqdm import tqdm
from flashtext import KeywordProcessor
from unidecode import unidecode

from preprocessing_pgp.address.scripts.utils import (
    number_pad_replace
)
from preprocessing_pgp.address.scripts.const import (
    DICT_NORM_ABBREV_REGEX_KW,
    DICT_NORM_CITY_DASH_REGEX
)


class VietnameseAddressCleaner:
    """
    Class support cleansing function for Vietnamese Addresses
    """

    # * PRIVATE
    def __replace_with_keywords(self, address: str, keywords: Dict) -> str:
        """
        Helper function to replace sub-address with given dictionary of keywords

        Parameters
        ----------
        address : str
            The input address to replace with keywords
        keywords : Dict
            The dictionary with `str` key - 'target' and (list `str`) as value - 'keywords'

        Returns
        -------
        str
            The replaced address
        """

        replaced_address = deepcopy(address)

        for replace_txt, target_subs in keywords.items():
            reg_target = '|'.join(target_subs)
            replaced_address = re.sub(reg_target,
                                      replace_txt,
                                      replaced_address)

            if replaced_address != address:
                return replaced_address

        return replaced_address

    def __clean_address_with_regex(self, address: str, regex: str) -> str:
        """
        Helper function to clean any address with given non-grouping regex & removing spaces
        """
        address_match = re.search(regex, address)

        if address_match is not None:
            sub_address = address_match.group(0).strip()
            cleaned_address = address.replace(
                sub_address,
                sub_address.replace(' ', '')
            )
        else:
            cleaned_address = address

        return cleaned_address

    def __clean_digit_district(self, address: str) -> str:
        """
        Helper function to clean district with digit

        * E.g: 'p 7' -> 'p7', and more
        """
        district_regex = r'[^A|a]p [0-9]+'

        cleaned_address = self.__clean_address_with_regex(
            address, district_regex)

        return cleaned_address

    def __clean_digit_ward(self, address: str) -> str:
        """
        Helper function to clean ward with digit

        * E.g: 'q 7' -> 'q7', and more
        """
        ward_regex = r'q [0-9]+'

        cleaned_address = self.__clean_address_with_regex(address, ward_regex)

        return cleaned_address

    # * PROTECTED
    def _unify_address(self, address: str) -> str:
        """
        Helper function to unify address to lower words and unidecode
        """
        unified_address = deepcopy(address)
        unified_address = unified_address.lower()
        unified_address = unidecode(unified_address)

        return unified_address

    def _remove_spare_spaces(self, address: str) -> str:
        """
        Helper function to remove spare spaces from string

        Parameters
        ----------
        address : str
            The input address to remove spare spaces

        Returns
        -------
        str
            Clean address without any spare spaces
        """

        cleaned_address = re.sub(' +', ' ', address)
        cleaned_address = cleaned_address.strip()

        return cleaned_address

    def _remove_padding_number(self, address: str) -> str:
        """
        Helper function to remove any number in string with padding zeros
        """

        digit_group_regex = r'(\d+)'

        cleaned_address = re.sub(digit_group_regex,
                                 number_pad_replace,
                                 address)

        return cleaned_address

    def _clean_digit_address(self, address: str) -> str:
        """
        Helper function to clean address containing digits

        Parameters
        ----------
        address : str
            The raw address which may contain digits

        Returns
        -------
        str
            Clean address with digits
        """
        cleaned_address = self.__clean_digit_district(address)
        cleaned_address = self.__clean_digit_ward(cleaned_address)

        return cleaned_address

    def _clean_dash_address(self, address: str) -> str:
        """
        Helper function to clean cities with dash '-'

        * E.g: 'ba ria vung tau' -> 'ba ria - vung tau', and more
        """

        return self.__replace_with_keywords(address, DICT_NORM_CITY_DASH_REGEX)

    def _clean_abbrev_address(self, address: str) -> str:
        """
        Helper function to clean & unify abbrev in address
        """
        return self.__replace_with_keywords(address, DICT_NORM_ABBREV_REGEX_KW)

    # * PUBLIC
    def clean_address(self, address: str) -> str:
        """
        Method for cleansing and unifying address

        Parameters
        ----------
        address : str
            The raw address that need cleansing and unifying

        Returns
        -------
        str
            Unified and cleaned address
        """
        unified_address = self._unify_address(address)

        cleaned_address = self._clean_abbrev_address(unified_address)

        cleaned_address = self._remove_spare_spaces(cleaned_address)

        cleaned_address = self._remove_padding_number(cleaned_address)

        cleaned_address = self._clean_digit_address(cleaned_address)

        cleaned_address = self._clean_dash_address(cleaned_address)

        return cleaned_address


def clean_vietnamese_address(address: str) -> str:
    """
    Function to clean and unify vietnamese address

    Parameters
    ----------
    address : str
        The raw address that need cleansing and unifying

    Returns
    -------
    str
        Final unified and cleansed address
    """
    cleaner = VietnameseAddressCleaner()

    cleaned_address = cleaner.clean_address(address)

    return cleaned_address
