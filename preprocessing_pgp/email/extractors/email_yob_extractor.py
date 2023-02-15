"""
Module to extract YOB from email using rule-based
"""
from itertools import product

import re
import pandas as pd
import numpy as np

from preprocessing_pgp.email.extractors.const import (
    DOB_FORMAT_DICT,
    DOB_REGEX_DICT,
    DOB_NUM_DIGIT_DICT
)


class EmailYOBExtractor:
    """
    Class contains logic to extract YOB from email
    """
    def __init__(self):
        self.yob_formats =\
            self.__generate_yob_dict()

    def __generate_yob_dict(self):
        date_list = [
            ['full_day', 'half_day', 'none'],  # DAY
            ['full_month', 'half_month', 'none'],  # MONTH
            ['full_year', 'half_year']  # YEAR
        ]

        possible_date_formats = list(product(*date_list))
        # Not possible if having day but not month
        possible_date_formats =\
            [date for date in possible_date_formats
             if (date[1] != 'none') | (date[0] == 'none')]
        # Remove case 4 number 2580 -> NOT YOB:1980
        possible_date_formats.remove(('half_day', 'half_month', 'half_year'))

        num_digits = []
        regexps = []
        formats = []

        for date_fm in possible_date_formats:
            date_num_digit = self.__get_date_digit(date_fm)
            date_regex = self.__get_date_regex(date_fm)
            date_format = self.__get_date_format(date_fm)

            num_digits.append(date_num_digit)
            regexps.append(date_regex)
            formats.append(date_format)

        return zip(num_digits, regexps, formats)

    # ? HELPER FUNCTION
    def __get_date_digit(
        self,
        date_fm
    ) -> int:
        return sum(DOB_NUM_DIGIT_DICT.get(fm, 0) for fm in date_fm)

    def __get_date_regex(
        self,
        date_fm
    ) -> str:
        return ''.join([DOB_REGEX_DICT.get(fm, '') for fm in date_fm])

    def __get_date_format(
        self,
        date_fm
    ) -> str:
        return ''.join(DOB_FORMAT_DICT.get(fm, '') for fm in date_fm)

    def _get_number_digit(
        self,
        string: str
    ) -> int:
        """
        Retrieve number of digit in string
        """
        return sum(c.isdigit() for c in string)

    def _extract_yob_regex(
        self,
        email_name: str,
        regex: str
    ) -> float:
        found_dates = re.findall(regex, email_name)

        if len(found_dates) == 0:
            return np.nan

    def _get_yob_with_format(
        self,
        email_name: str,
        yob_fm
    ) -> float:
        """
        Extract yob in email name with specific format
        """
        num_digit, regexp, _ = yob_fm

        # * Check for number of digit
        num_digit_name = self._get_number_digit(email_name)
        if num_digit_name != num_digit:
            return np.nan

        patterns = re.findall(regexp, email_name)

        if len(patterns) == 0:
            return np.nan

        # Case only year
        if isinstance(patterns[-1], str):
            yob = float(patterns[-1])
        else:
            yob = float(patterns[-1][-1])

        # Edge case 2K birth
        if yob == 20.0:
            return 2000.0

        # 1980 -> 1980, 8 -> 2008, 98 -> 1998
        yob = 1900+yob if 9<yob<100 else 2000+yob if yob <= 9 else yob

        return yob

    def _get_yob(
        self,
        data: pd.DataFrame,
        email_name_col: str = 'email_name'
    ) -> pd.DataFrame:
        """
        Generate an yob_extracted columns
        """
        data['yob_extracted'] = np.nan
        for yob_fm in self.yob_formats:
            not_found_mask = data['yob_extracted'].isna()
            data.loc[
                not_found_mask,
                'yob_extracted'
            ] = data.loc[
                not_found_mask,
                email_name_col
            ].apply(
                lambda email_name, fm=yob_fm:
                self._get_yob_with_format(email_name, fm)
            )

        return data


    # ? MAIN FUNCTION
    def extract_yob(
        self,
        data: pd.DataFrame,
        email_name_col: str = 'email_name'
    ) -> pd.DataFrame:
        """
        Extract YOB from email name

                Parameters
        ----------
        data : pd.DataFrame
            Data containing the email
        email_name_col : str, optional
            Name of email column from data, by default 'email_name'

        Returns
        -------
        pd.DataFrame
            Data with additional columns:
            * `yob_extracted` : The yob extracted from email
        """
        # * Extracting yob
        extracted_data = self._get_yob(
            data,
            email_name_col
        )

        return extracted_data
