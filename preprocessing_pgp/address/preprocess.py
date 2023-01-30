"""
This module meant for preprocessing (cleansing and unifying) the address before putting into extraction
"""

import re
from typing import Dict
from copy import deepcopy
from string import punctuation

import pandas as pd
from unidecode import unidecode
from halo import Halo

from preprocessing_pgp.address.utils import (
    number_pad_replace
)
from preprocessing_pgp.name.preprocess import (
    remove_spare_spaces,
)
from preprocessing_pgp.address.const import (
    DICT_NORM_ABBREV_REGEX_KW,
    DICT_NORM_CITY_DASH_REGEX,
    ADDRESS_PUNCTUATIONS
)


class VietnameseAddressCleaner:
    """
    Class support cleansing function for Vietnamese Addresses
    """
    non_address_punctuation = ''.join([pun for pun in punctuation
                                       if pun not in ADDRESS_PUNCTUATIONS])

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
            reg_target = re.compile('|'.join(map(re.escape, target_subs)))
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

    def _clean_full_address(self, address: str) -> str:
        """
        Method for cleansing full address removing special characters

        Parameters
        ----------
        address : str
            The remained address to be cleaned

        Returns
        -------
        str
            The cleaned address
        """
        clean_address =\
            address.translate(str.maketrans(
                '', '', self.non_address_punctuation))

        clean_address = remove_spare_spaces(clean_address)

        return clean_address

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

        cleaned_address = self._clean_full_address(cleaned_address)

        return cleaned_address


@Halo(
    text='Cleansing address',
    color='cyan',
    spinner='dots7',
    text_color='magenta'
)
def clean_vi_address(
    data: pd.DataFrame,
    address_col: str
) -> pd.DataFrame:
    """
    Function to clean and unify vietnamese address in data

    Parameters
    ----------
    data : pd.DataFrame
        Raw data containing the address
    address_col : str
        The raw address column that need cleansing and unifying

    Returns
    -------
    pd.DataFrame
        Final unified and cleansed data with new column named `cleaned_<address_col>`
    """
    cleaner = VietnameseAddressCleaner()

    cleaned_data = data.copy()

    cleaned_data[f'cleaned_{address_col}'] =\
        cleaned_data[address_col].apply(
            cleaner.clean_address
    )

    return cleaned_data
