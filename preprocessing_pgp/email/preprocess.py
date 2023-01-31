"""
Module to preprocess email before proceed more
"""

import re
import pandas as pd
from halo import Halo
from preprocessing_pgp.email.const import NAN_EMAIL_LIST


class EmailCleaner:
    """
    Class support cleansing function for Email
    """

    def clean_email(
        self,
        email: str
    ) -> str:
        """
        Process cleaning email:
        1. Remove spaces

        Parameters
        ----------
        email : str
            The input email to clean

        Returns
        -------
        str
            Cleaned email without any spaces
        """

        # Case NaN email
        if email is None or email in NAN_EMAIL_LIST:
            return None

        cleaned_email = email.lower()
        cleaned_email = self._remove_spaces(cleaned_email)

        return cleaned_email

    def _remove_spaces(
        self,
        email: str
    ) -> str:
        """
        Function to remove spaces from email
        """
        cleaned_email = re.sub(' +', '', email)

        return cleaned_email


@Halo(
    text='Cleansing email',
    color='cyan',
    spinner='dots7',
    text_color='magenta'
)
def clean_email(
    data: pd.DataFrame,
    email_col: str = 'email'
) -> pd.DataFrame:
    """
    Process cleansing email from dataframe
    and created `cleaned_<email_col>` column contains the cleaned email

    Parameters
    ----------
    data : pd.DataFrame
        The input dataframe contains a column with email
    email_col : str, optional
        The name of the column contains email records, by default 'email'

    Returns
    -------
    pd.DataFrame
        The original dataframe with an additional column contains cleaned email
    """

    cleaner = EmailCleaner()

    cleaned_data = data

    cleaned_data[f'cleaned_{email_col}'] =\
        cleaned_data[email_col].apply(
            cleaner.clean_email
    )

    return cleaned_data
