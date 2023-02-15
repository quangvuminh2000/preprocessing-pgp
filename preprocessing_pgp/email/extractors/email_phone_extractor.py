"""
Module to extract phone from email using rule-based
"""

import re
import pandas as pd

from preprocessing_pgp.email.extractors.const import VIETNAMESE_PHONE_REGEX


class EmailPhoneExtractor:
    """
    Class contains logic to extract phone number from email
    """

    def _get_phone(
        self,
        email_name: str
    ) -> str:
        """
        Extract phone if exist from email name
        """

        phone_match = re.findall(VIETNAMESE_PHONE_REGEX, email_name)

        if len(phone_match) == 0:
            return None

        return phone_match[0][0]

    def extract_phone(
        self,
        data: pd.DataFrame,
        email_name_col: str = 'email_name'
    ) -> pd.DataFrame:
        """
        Extract phone number from email name if possible

        Parameters
        ----------
        data : pd.DataFrame
            The input data contains an email_name column
        email_name_col : str, optional
            The name of the column contains the email name, by default 'email_name'

        Returns
        -------
        pd.DataFrame
            Data with additional column for extracted phone:
            * `phone_extracted` : Extracted phone from email name
        """

        # * Using regex to search for phone
        data['phone_extracted'] =\
            data[email_name_col].apply(self._get_phone)

        return data
