"""
Module contains validator for email
"""

import re
from time import time

import pandas as pd
from halo import Halo

from preprocessing_pgp.email.utils import (
    split_email,
    is_name_accented
)
from preprocessing_pgp.email.const import (
    AT_LEAST_ONE_CHAR_REGEX,
    LEAST_NUM_EMAIL_CHAR,
    EMAIL_DOMAIN_REGEX,
    COMMON_EMAIL_REGEX,
    EDGE_AUTO_EMAIL_REGEX,
    PRIVATE_EMAIL_DOMAINS,
    DOMAIN_GROUP_DICT
)
from preprocessing_pgp.email.preprocess import (
    clean_email
)
from preprocessing_pgp.utils import (
    sep_display,
    parallelize_dataframe
)


class EmailValidator:
    """
    Class contains building components for validating email
    """

    def __init__(self) -> None:
        self.email_services = EMAIL_DOMAIN_REGEX.values()

    def is_valid_email(
        self,
        email: str = None
    ) -> bool:
        """
        Check whether the email is valid or not

        Parameters
        ----------
        email : str
            The input email to check for validation

        Returns
        -------
        bool
            Whether the input email is valid with the basic email rules
        """

        if not email:
            return False

        normed_email = email.lower()

        if self.is_auto_email(normed_email):
            return False

        return self.is_large_company_email(normed_email)\
            or self.is_common_email(normed_email)\
            or self.is_student_email(normed_email)

    def is_large_company_email(
        self,
        email: str
    ) -> bool:
        """
        Check whether the email is of large company email:
        1. gmail
        2. yahoo
        3. microsoft
        4. fpt

        Parameters
        ----------
        email : str
            The input email to validate for large company email

        Returns
        -------
        bool
            Whether the email if valid large company email
        """
        for email_service in self.email_services:
            _, email_group = split_email(email)
            if email_group in email_service['domains']:
                return bool(re.match(email_service['regex'], email))

        return False

    def is_common_email(
        self,
        email: str
    ) -> bool:
        """
        Check whether the email is common email account or not
        """

        if re.match(COMMON_EMAIL_REGEX, email):
            email_name, _ = split_email(email)
            return self._is_valid_email_name(email_name)
        return False

    def is_student_email(
        self,
        email: str
    ) -> bool:
        """
        Check whether the email is of student email or not
        """
        email_name, email_group = split_email(email)

        if not email_group:
            return False

        if re.search('edu', email_group):
            return self._is_valid_email_name(email_name)

        return False

    def is_auto_email(
        self,
        email: str
    ) -> bool:
        """
        Check whether the email is auto-email or not
        """
        return bool(re.match(EDGE_AUTO_EMAIL_REGEX, email))

    def _is_valid_email_name(
        self,
        email_name: str
    ) -> bool:
        """
        Check valid email's name with 3 criteria:
        1. Non-accented
        2. At least 8 characters
        3. Contains at least 1 alpha character

        Parameters
        ----------
        email_name : str
            The input email's name

        Returns
        -------
        bool
            Whether the email name is valid or not
        """

        return not is_name_accented(email_name)\
            and self.__is_valid_name_length(email_name)\
            and self.__is_valid_name_syntax(email_name)

    def __is_valid_name_length(
        self,
        email_name: str
    ) -> bool:
        """
        Check whether the email's name length is in valid range
        """
        return len(email_name) >= LEAST_NUM_EMAIL_CHAR

    def __is_valid_name_syntax(
        self,
        email_name: str
    ) -> bool:
        """
        Check whether the email's name is at valid syntax
        """
        return bool(re.match(AT_LEAST_ONE_CHAR_REGEX, email_name))


@Halo(
    text='Validating email',
    color='cyan',
    spinner='dots7',
    text_color='magenta'
)
def validate_clean_email(
    data: pd.DataFrame,
    email_col: str = 'cleaned_email'
) -> pd.DataFrame:
    """
    Process validating for cleaned email
    and output an additional column for validator: `is_email_valid`

    Parameters
    ----------
    data : pd.DataFrame
        The dataframe contains cleaned email column
    email_col : str, optional
        The cleaned email column name, by default 'cleaned_email'

    Returns
    -------
    pd.DataFrame
        Validated data contains the email's valid indicator: `is_email_valid`
    """

    validator = EmailValidator()

    validated_data = data.copy()

    validated_data['is_email_valid'] = validated_data[email_col].apply(
        validator.is_valid_email
    )
    # * Check for autoemail
    validated_data['is_autoemail'] = validated_data[email_col].apply(
        validator.is_auto_email
    )

    return validated_data


def process_validate_email(
    data: pd.DataFrame,
    email_col: str = 'email',
    n_cores: int = 1
) -> pd.DataFrame:
    """
    Process validating email address

    Parameters
    ----------
    data : pd.DataFrame
        The dataframe contains email records
    email_col : str, optional
        The column name that hold email records, by default 'email'
    n_cores : int, optional
        The number of cores used to run parallel, by default 1 core will be used

    Returns
    -------
    pd.DataFrame
        The data with additional columns:
        * `is_email_valid`: indicator for whether the email is valid or not
        * `is_autoemail`: indicator for whether the email is autoemail or not
    """
    # * Select only email column to continue
    orig_cols = data.columns
    email_data = data[[email_col]]

    # * Cleansing email
    start_time = time()
    if n_cores == 1:
        cleaned_data = clean_email(
            email_data,
            email_col=email_col
        )
    else:
        cleaned_data = parallelize_dataframe(
            email_data,
            clean_email,
            n_cores=n_cores,
            email_col=email_col
        )
    clean_time = time() - start_time
    print(f"Cleansing email takes {int(clean_time)//60}m{int(clean_time)%60}s")
    sep_display()

    # * Separate na data
    na_data = cleaned_data[cleaned_data[f'cleaned_{email_col}'].isna()]
    non_na_data = cleaned_data[cleaned_data[f'cleaned_{email_col}'].notna()]

    # * Validating email
    start_time = time()
    if n_cores == 1:
        validated_data = validate_clean_email(
            non_na_data,
            email_col=f'cleaned_{email_col}'
        )
    else:
        validated_data = parallelize_dataframe(
            non_na_data,
            validate_clean_email,
            n_cores=n_cores,
            email_col=f'cleaned_{email_col}'
        )
    validate_time = time() - start_time
    print(
        f"Validating email takes {int(validate_time)//60}m{int(validate_time)%60}s")
    sep_display()

    # * Get the domain of the email name & Check for private email
    validated_data['email_domain'] = validated_data[email_col].str.split('@').str[1]
    validated_data['email_domain'] = validated_data['email_domain'].replace(DOMAIN_GROUP_DICT)
    validated_data['private_email'] = validated_data['email_domain'].isin(PRIVATE_EMAIL_DOMAINS)

    # * Concat with the nan data
    final_data = pd.concat([validated_data, na_data])

    # * Filling na data to invalid email
    final_data['is_email_valid'].fillna(False, inplace=True)

    # * Concat with the origin cols
    new_cols = [
        f'cleaned_{email_col}',
        'is_email_valid',
        'is_autoemail',
        'email_domain',
        'private_email'
    ]
    final_data = pd.concat([data[orig_cols], final_data[new_cols]], axis=1)

    return final_data
