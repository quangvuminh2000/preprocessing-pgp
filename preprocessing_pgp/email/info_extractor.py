"""
Module contains objects and functions to support extracting information from email
"""

from time import time

import pandas as pd
from halo import Halo

from preprocessing_pgp.utils import parallelize_dataframe, sep_display
from preprocessing_pgp.email.validator import process_validate_email
from preprocessing_pgp.email.extractors.email_name_extractor import EmailNameExtractor


class EmailInfoExtractor:
    def __init__(self):
        self.name_extractor = EmailNameExtractor()

    @Halo(
        text='Extracting information from email',
        color='cyan',
        spinner='dots7',
        text_color='magenta'
    )
    def extract_info(
        self,
        data: pd.DataFrame,
        email_name_col:str = 'email_name'
    ) -> pd.DataFrame:
        # * Extracting name
        extracted_data =\
            self.name_extractor.extract_username(data, email_name_col)

        return extracted_data


def process_extract_email_info(
    data: pd.DataFrame,
    email_col:str = 'email',
    n_cores:int =1
) -> pd.DataFrame:
    """
    Process extracting information from email, extracted information may conclude:
    1. Username -- with accent
    2. Customer type -- derived from username
    3. Year of birth
    4. Address -- 3 levels
    5. Email group
    6. Auto-email

    Parameters
    ----------
    data : pd.DataFrame
        The input data contains an email column
    email_col : str
        The name of the column contains email's records, by default `email`
    n_cores

    Returns
    -------
    pd.DataFrame
        Original data with additional columns contains 6 additional information as above
    """

    info_extractor = EmailInfoExtractor()
    # ? Validate email -- Only extract info for valid email
    validated_data = process_validate_email(
        data,
        email_col=email_col,
        n_cores=n_cores
    )
    valid_email = validated_data.query('is_email_valid').copy()
    invalid_email = validated_data.query('~is_email_valid').copy()
    # ? Separate email name and group
    valid_email[f'{email_col}_name'] = valid_email[email_col].str.split('@').str[0]

    # ? Extract username from email
    start_time = time()
    extracted_valid_email = parallelize_dataframe(
        valid_email,
        info_extractor.extract_info,
        n_cores=n_cores,
        email_name_col=f'{email_col}_name'
    )
    extract_time = time() - start_time
    print(
        f"Extracting information from email takes {int(extract_time)//60}m{int(extract_time)%60}s")
    sep_display()

    final_data = pd.concat([extracted_valid_email, invalid_email])

    return final_data
