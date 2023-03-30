"""
Module contains validator for email
"""

import re
from time import time

import pandas as pd

from preprocessing_pgp.email.utils import (
    split_email,
    is_name_accented,
    is_valid_email_domain
)
from preprocessing_pgp.email.const import (
    AT_LEAST_ONE_CHAR_REGEX,
    LEAST_NUM_EMAIL_CHAR,
    EMAIL_DOMAIN_REGEX,
    COMMON_EMAIL_REGEX,
    EDGE_AUTO_EMAIL_REGEX,
    PRIVATE_EMAIL_DOMAINS,
    DOMAIN_GROUP_DICT,
    EDU_EMAIL_REGEX
)
from preprocessing_pgp.email.preprocess import (
    clean_email
)
from preprocessing_pgp.utils import (
    parallelize_dataframe
)

pd.options.mode.chained_assignment = None


class EmailValidator:
    """
    Class contains building components for validating email
    """

    def __init__(
        self,
        domain_dict: pd.DataFrame = None
    ) -> None:
        self.email_services = EMAIL_DOMAIN_REGEX.values()
        self.domain_dict = domain_dict

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
        email_name, email_group = split_email(email)

        # if not is_valid_email_domain(email_group):
        #     return False

        if self.is_auto_email(normed_email):
            return False

        large_company_domain = self.get_large_company_domain(email_group)
        if large_company_domain is not None:
            return self._is_valid_email_regex(
                email_name,
                EMAIL_DOMAIN_REGEX[large_company_domain]['regex']
            )

        if self.is_student_email(email_group):
            return self._is_valid_email_regex(
                email_name,
                EDU_EMAIL_REGEX
            )

        return self.is_common_email(normed_email)

    def _is_valid_email_regex(
        self,
        email: str,
        email_regex: str
    ) -> bool:
        """
        Check whether the email match the regex
        """
        return bool(re.match(email_regex, email))

    def get_large_company_domain(
        self,
        email_group: str
    ) -> str:
        """
        Check whether the email is of large company email:
        1. gmail
        2. yahoo
        3. microsoft
        4. fpt

        Parameters
        ----------
        email_group : str
            The input email_group to get the domain name

        Returns
        -------
        str
            The email group that email was in
        """
        for domain_regex, domain_name in DOMAIN_GROUP_DICT.items():
            if re.match(domain_regex, email_group):
                return domain_name

        return None

    def is_common_email(
        self,
        email: str
    ) -> bool:
        """
        Check whether the email is common email account or not
        """

        if re.match(COMMON_EMAIL_REGEX, email):
            # email_name, _ = split_email(email)
            # return self._is_valid_email_name(email_name)
            return True
        return False

    def is_student_email(
        self,
        email_group: str
    ) -> bool:
        """
        Check whether the email is of student email or not
        """
        if not email_group:
            return False

        return bool(re.search('edu', email_group))

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

    def check_valid_email_domain(
        self,
        data: pd.DataFrame,
        domain_col: str = 'email_domain'
    ) -> pd.DataFrame:
        """
        Checking whether the email domain is valid or not
        """
        if self.domain_dict is None:
            data['is_domain_valid'] =\
                data[domain_col].apply(is_valid_email_domain)
            return data

        if not self.domain_dict.columns.isin(['email_domain', 'is_domain_valid']).all():
            print("Domain dictionary not match schema, re-running all domains")
            data['is_domain_valid'] =\
                data[domain_col].apply(is_valid_email_domain)
            return data

        data_domains = data[data[domain_col].notna()][domain_col].unique()
        new_domains = list(set(data_domains)-set(self.domain_dict['email_domain'].unique()))
        new_domain_df = pd.DataFrame({
            'email_domain': new_domains
        })
        new_domain_df['is_domain_valid'] \
            = new_domain_df['email_domain'].apply(is_valid_email_domain)

        domain_data = pd.concat([
            self.domain_dict.drop_duplicates(subset='email_domain', keep='last'),
            new_domain_df
        ], ignore_index=True)
        domain_mapper = dict(zip(
            domain_data['email_domain'],
            domain_data['is_domain_valid']
        ))
        data['is_domain_valid'] = data[domain_col].map(domain_mapper)
        data['is_domain_valid'] = data['is_domain_valid'].fillna(False)
        data['is_domain_valid'] = data['is_domain_valid'].astype(bool)

        return data


def validate_clean_email(
    data: pd.DataFrame,
    email_col: str = 'cleaned_email',
    domain_dict: pd.DataFrame = None
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
    domain_dict : pd.DataFrame, optional
        The domain dictionary to use for faster reference, by default None

    Returns
    -------
    pd.DataFrame
        Validated data contains the email's valid indicator: `is_email_valid`
    """
    if data.empty:
        return data

    validator = EmailValidator(domain_dict=domain_dict)

    validated_data = data

    # * Check for email domain
    validated_data['email_domain'] =\
        validated_data[email_col]\
        .str.split('@').str[1]
    validated_data = validator.check_valid_email_domain(
        validated_data,
        domain_col='email_domain'
    )
    valid_domain_data = validated_data.query('is_domain_valid')
    invalid_domain_data = validated_data.query('~is_domain_valid')

    # * Check for email regex
    valid_domain_data['is_email_valid'] = valid_domain_data[email_col].apply(
        validator.is_valid_email
    )

    # * Concat with invalid domain data
    invalid_domain_data['is_email_valid'] = False
    validated_data = pd.concat([valid_domain_data, invalid_domain_data])

    # * Check for autoemail
    validated_data['is_autoemail'] = validated_data[email_col].apply(
        validator.is_auto_email
    )

    return validated_data


def process_validate_email(
    data: pd.DataFrame,
    email_col: str = 'email',
    n_cores: int = 1,
    domain_dict: pd.DataFrame = None
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
    domain_dict : pd.DataFrame, optional
        The domain dictionary to use for faster reference, by default None

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
    print(">>> Cleansing email: ", end='')
    start_time = time()
    cleaned_data = parallelize_dataframe(
        email_data,
        clean_email,
        n_cores=n_cores,
        email_col=email_col
    )
    clean_time = time() - start_time
    print(f"{int(clean_time)//60}m{int(clean_time)%60}s")

    # * Separate na data
    na_data = cleaned_data[cleaned_data[f'cleaned_{email_col}'].isna()]
    non_na_data = cleaned_data[cleaned_data[f'cleaned_{email_col}'].notna()]

    # * Validating email
    print(">>> Validating email: ", end='')
    start_time = time()
    validated_data = parallelize_dataframe(
        non_na_data,
        validate_clean_email,
        n_cores=n_cores,
        email_col=f'cleaned_{email_col}',
        domain_dict=domain_dict
    )
    validate_time = time() - start_time
    print(f"{int(validate_time)//60}m{int(validate_time)%60}s")

    # * Get the domain of the email name & Check for private email
    print(">>> Get email domain: ", end='')
    start_time = time()
    validated_data['email_domain'] = validated_data['email_domain'].replace(
        DOMAIN_GROUP_DICT, regex=True)
    validated_data['private_email'] = validated_data['email_domain'].isin(
        PRIVATE_EMAIL_DOMAINS)
    domain_time = time() - start_time
    print(f"{int(domain_time)//60}m{int(domain_time)%60}s")

    # * Concat with the nan data
    final_data = pd.concat([validated_data, na_data])

    # * Filling na data to invalid email
    final_data['is_email_valid'].fillna(False, inplace=True)
    final_data['is_email_valid'] = final_data['is_email_valid'].astype(bool)

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
