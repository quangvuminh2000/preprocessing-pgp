import os
import re
from abc import ABC, abstractmethod
from time import time

import pandas as pd
from preprocessing_pgp.card.const import (  # Personal ID; Passport; Driver License
    DRIVER_LICENSE_ID_REGION_CODES,
    DRIVER_LICENSE_LENGTH,
    GENDER_NUM_TO_CENTURY,
    INVALID_DRIVER_LICENSE_FIRST_YEAR_CHAR,
    INVALID_DRIVER_LICENSE_PASSING_YEAR,
    NEW_PID_CODE_LENGTH,
    NEW_PID_REGION_CODE_NUMS,
    OLD_PID_CODE_LENGTH,
    OLD_PID_REGION_CODE_NUMS,
    PASSPORT_LENGTH,
    PASSPORT_PATTERN,
    POSSIBLE_GENDER_NUM,
    REGION_CODE_DICT,
    VALID_DRIVER_LICENSE_LAST_YEAR_CHAR,
    VALID_PID_21_CENTURY_DOB,
)
from preprocessing_pgp.card.preprocess import clean_card_data
from preprocessing_pgp.card.utils import is_checker_valid
from preprocessing_pgp.utils import (  # apply_multi_process,; apply_progress_bar,
    extract_null_values,
    parallelize_dataframe,
    sep_display,
)
from tqdm import tqdm

tqdm.pandas()
PROCESSES = os.cpu_count() // 2


class CardValidator(ABC):
    """
    Abstract Class contains validating functions for validating card_id
    """

    @staticmethod
    def is_valid_gender(gender_code: str) -> bool:
        """
        Check if the gender code is of correct format

        Parameters
        ----------
        gender_code : str
            Gender code extracted from card_id

        Returns
        -------
        bool
            Whether the gender code is at valid form
        """
        return gender_code in POSSIBLE_GENDER_NUM

    @staticmethod
    def is_all_digit_card(card_id: str) -> bool:
        """
        Check if the card_id contains all digit or not

        Parameters
        ----------
        card_id : str
            The card id

        Returns
        -------
        bool
            Whether the card id contains all digit or not
        """
        return card_id.isdecimal()

    @staticmethod
    @abstractmethod
    def is_valid_card(card_id: str) -> bool:
        """
        Check if the card is valid by syntax or not

        Parameters
        ----------
        card_id : str
            The input card id

        Returns
        -------
        bool
            Whether the card is valid by syntax
        """
        raise NotImplementedError("Subclasses should implement this!")


class PersonalIDValidator(CardValidator):
    """
    Class to check for personal id syntax
    """

    @staticmethod
    def is_old_card(card_id: str) -> bool:
        """
        Helper function to validate old card length
        """
        return len(card_id) == OLD_PID_CODE_LENGTH

    @staticmethod
    def is_new_card(card_id: str) -> bool:
        """
        Helper function to validate new card length
        """
        return len(card_id) == NEW_PID_CODE_LENGTH

    @staticmethod
    def is_valid_old_card(card_id: str) -> bool:
        """
        Helper function to check if the card_id is of old card type
        """
        if not PersonalIDValidator.is_old_card(card_id):
            return False
        return any(
            (
                card in OLD_PID_REGION_CODE_NUMS
                for card in [card_id[:2], card_id[:3]]
            )
        )

    @staticmethod
    def is_valid_new_card(card_id: str) -> bool:
        """
        Helper function to check if the card_id is of new card type

        New card type contains:
        1.
        """
        if not PersonalIDValidator.is_new_card(card_id):
            return False

        if card_id[:3] in NEW_PID_REGION_CODE_NUMS:
            gender_code = card_id[3]
            return CardValidator.is_valid_gender(
                gender_code
            ) and PersonalIDValidator.is_valid_range_birth(card_id)

        return False

    @staticmethod
    def is_semi_correct_length(card_id: str) -> bool:
        """
        Helper function to check if the card_id length is still acceptable
        """
        return len(card_id) in (
            OLD_PID_CODE_LENGTH - 1,
            NEW_PID_CODE_LENGTH - 1,
        )

    @staticmethod
    def is_valid_semi_card(card_id: str) -> bool:
        """
        Helper function to check if the card_id is of semi-correct length and is correct

        Steps
        -----

        1. Check the length of the card id
        2. Modify card id by adding '0' at the beginning and check for whether it is valid or not
        """
        if not PersonalIDValidator.is_semi_correct_length(card_id):
            return False

        modified_card_id = "0" + card_id
        return PersonalIDValidator.is_valid_old_card(
            modified_card_id
        ) or PersonalIDValidator.is_valid_new_card(modified_card_id)

    @staticmethod
    def is_valid_range_birth(card_id: str) -> bool:
        """
        Helper to check whether the dob in new card_id is in valid range or not
        """
        gender_code = card_id[3]
        birth_year = card_id[4:6]

        # Century 21
        if gender_code in GENDER_NUM_TO_CENTURY["21"]:
            return birth_year in VALID_PID_21_CENTURY_DOB

        return True

    @staticmethod
    def is_valid_card(card_id: str) -> bool:
        """
        Check if the card is valid by personal id syntax or not

        The valid card must be `all digit` and `valid syntax` of old or new card

        Parameters
        ----------
        card_id : str
            The input card id

        Returns
        -------
        bool
            Whether the card is valid by personal id syntax
        """
        if not CardValidator.is_all_digit_card(card_id):
            return False

        return (
            PersonalIDValidator.is_valid_new_card(card_id)
            or PersonalIDValidator.is_valid_old_card(card_id)
            or PersonalIDValidator.is_valid_semi_card(card_id)
        )


class PassportValidator(CardValidator):
    """
    Class to check for passport id syntax
    """

    @staticmethod
    def is_valid_length(card_id: str) -> bool:
        """
        Helper function to validate the correct length of the card
        """
        return len(card_id) == PASSPORT_LENGTH

    @staticmethod
    def is_valid_syntax(card_id: str) -> bool:
        """
        Helper function to validate passport syntax:

        Steps
        -----

        1. Contains a starting character
        2. Next is 7 random digits
        """
        return bool(re.match(PASSPORT_PATTERN, card_id.lower()))

    @staticmethod
    def is_valid_card(card_id: str) -> bool:
        """
        Check if the card is valid by passport syntax or not

        Parameters
        ----------
        card_id : str
            The input card id

        Returns
        -------
        bool
            Whether the card is valid by passport syntax
        """
        return PassportValidator.is_valid_length(
            card_id
        ) and PassportValidator.is_valid_syntax(card_id)


class DriverLicenseValidator(CardValidator):
    """
    Class to check for driver license syntax
    """

    @staticmethod
    def is_valid_length(card_id: str) -> bool:
        """
        Helper function to validate the correct length of the card
        """
        return len(card_id) == DRIVER_LICENSE_LENGTH

    @staticmethod
    def is_valid_region_code(card_id: str) -> bool:
        """
        Function to check for valid region code in driver license card id
        """
        region_code = card_id[:2]

        return region_code in DRIVER_LICENSE_ID_REGION_CODES

    @staticmethod
    def is_valid_passing_year(card_id: str) -> bool:
        """
        Helper function to validate driver license passing year in card:
        """
        passing_year = card_id[3:5]

        return passing_year not in INVALID_DRIVER_LICENSE_PASSING_YEAR

    @staticmethod
    def is_valid_gender_code(card_id: str) -> bool:
        """
        Helper function to validate driver license passing year in card:
        """
        gender_code = card_id[2]

        return CardValidator.is_valid_gender(gender_code)

    @staticmethod
    def is_real_driver_license(card_id: str) -> bool:
        """
        Helper function to validate the card id
        in the case of the first 3 digits is in personal id region code
        """
        if PersonalIDValidator.is_valid_new_card(card_id):
            first_year_char = card_id[3]
            second_year_char = card_id[4]

            return (
                first_year_char not in INVALID_DRIVER_LICENSE_FIRST_YEAR_CHAR
            ) and (second_year_char in VALID_DRIVER_LICENSE_LAST_YEAR_CHAR)

        return True

    @staticmethod
    def is_valid_card(card_id: str) -> bool:
        """
        Check if the card is valid by driver license syntax or not

        Parameters
        ----------
        card_id : str
            The input card id

        Returns
        -------
        bool
            Whether the card is valid by passport syntax
        """
        if not CardValidator.is_all_digit_card(card_id):
            return False

        return (
            DriverLicenseValidator.is_valid_length(card_id)
            and DriverLicenseValidator.is_valid_region_code(card_id)
            and DriverLicenseValidator.is_valid_gender_code(card_id)
            and DriverLicenseValidator.is_valid_passing_year(card_id)
            and DriverLicenseValidator.is_real_driver_license(card_id)
        )


def parse_gender_from_card(
    profile: pd.DataFrame,
    card_col: str = "card_id",
) -> pd.DataFrame:
    """
    Parsing gender from new pid and driver license

    Parameters
    ----------
    profile : pd.DataFrame
        The input DF containing card id
    card_col : str, optional
        The column name that contain card data, by default 'card_id'

    Returns
    -------
    pd.DataFrame
        Data with extra info:
        * `gender`: Gender extracted from new pid and driver license
    """
    gender_dict = {"0": "M", "1": "F", "2": "M", "3": "F"}
    # * Gender from new personal's id
    new_pid_mask = (profile["is_personal_id"]) & (
        profile[card_col].apply(PersonalIDValidator.is_new_card)
    )

    # * Process for semi-card
    semi_card_mask = profile[card_col].apply(
        PersonalIDValidator.is_semi_correct_length
    )
    profile.loc[new_pid_mask & semi_card_mask, card_col] = (
        "0" + profile[card_col]
    )

    # * Map gender by dict
    profile["gender"] = None
    profile.loc[new_pid_mask, "gender"] = (
        profile[card_col].str[3].map(gender_dict)
    )

    # * Gender from driver license
    driver_license_mask = (profile["is_driver_license"]) & (
        profile["gender"].notna()
    )

    profile.loc[driver_license_mask, "gender"] = (
        profile[card_col].str[2].map(gender_dict)
    )

    return profile


def parse_birthday_from_card(
    profile: pd.DataFrame,
    card_col: str = "card_id",
) -> pd.DataFrame:
    """
    Parsing birthday from new personal id

    Parameters
    ----------
    profile : pd.DataFrame
        The input DF containing card id
    card_col : str, optional
        The column name that contain card data, by default 'card_id'

    Returns
    -------
    pd.DataFrame
        Data with extra info:
        * `year_of_birth`: YOB extracted from new pid
    """
    # * Can only get birthday from new personal's id
    new_pid_mask = (profile["is_personal_id"]) & (
        profile[card_col].apply(PersonalIDValidator.is_new_card)
    )

    # * Process for semi-card
    semi_card_mask = profile[card_col].apply(
        PersonalIDValidator.is_semi_correct_length
    )
    profile.loc[new_pid_mask & semi_card_mask, card_col] = (
        "0" + profile[card_col]
    )

    gender_post_century = {"0": "19", "1": "19", "2": "20", "3": "20"}
    profile.loc[new_pid_mask, "post_cent"] = (
        profile[card_col].str[3].map(gender_post_century)
    )

    profile.loc[new_pid_mask, "year_of_birth"] = (
        profile["post_cent"] + profile[card_col].str[4:6]
    )

    profile = profile.drop(columns=["post_cent"])

    return profile


def parse_city_from_card(
    profile: pd.DataFrame,
    card_col: str = "card_id",
) -> pd.DataFrame:
    """
    Parsing city from new personal id and driver license

    Parameters
    ----------
    profile : pd.DataFrame
        The input DF containing card id
    card_col : str, optional
        The column name that contain card data, by default 'card_id'

    Returns
    -------
    pd.DataFrame
        Data with extra info:
        * `city`: City extracted from new pid and driver license
    """
    # * New personal ID
    new_pid_mask = (profile["is_personal_id"]) & (
        profile[card_col].apply(PersonalIDValidator.is_new_card)
    )
    profile.loc[new_pid_mask, "code"] = profile[card_col].str[0:3]

    # * Process for semi-card
    semi_card_mask = profile[card_col].apply(
        PersonalIDValidator.is_semi_correct_length
    )
    profile.loc[new_pid_mask & semi_card_mask, card_col] = (
        "0" + profile[card_col]
    )

    # * Driver License
    driver_license_mask = (profile["is_driver_license"]) & (
        ~profile["is_personal_id"]
    )
    profile.loc[driver_license_mask, "code"] = profile[card_col].str[0:2]

    # * Merge to get city
    profile = (
        pd.merge(
            profile.set_index("code"),
            REGION_CODE_DICT.set_index("code"),
            left_index=True,
            right_index=True,
            how="left",
            sort=False,
        )
        .reset_index()
        .drop(columns=["code"])
    )

    return profile


def extract_card_info(
    profile: pd.DataFrame, card_col: str = "card_id"
) -> pd.DataFrame:
    """
    Parsing information from valid card id

    Parameters
    ----------
    profile : pd.DataFrame
        The input DF containing card id
    card_col : str, optional
        The column name that contain card data, by default 'card_id'

    Returns
    -------
    pd.DataFrame
        Data include extra info:
        * `gender`: Gender parsed
        * `year_of_birth`: YOB parsed
        * `city`: City parsed
    """
    if profile.empty:
        return profile

    # * Parsing gender from card id
    verified_data = parse_gender_from_card(profile, card_col=card_col)

    # * Parsing birthday from card id
    verified_data = parse_birthday_from_card(verified_data, card_col=card_col)

    # * Parsing city from card_id
    verified_data = parse_city_from_card(verified_data, card_col=card_col)

    return verified_data


def verify_card(
    data: pd.DataFrame, card_col: str = "card_id", print_info: bool = True
) -> pd.DataFrame:
    """
    Verify whether the card ids are valid or not

    Parameters
    ----------
    card_df : pd.DataFrame
        The input DF containing card id
    card_col : str, optional
        The column contain card id, by default "card_id"
    print_info : bool, optional
        Whether to print the information of the run, by default True

    Returns
    -------
    pd.DataFrame
        The final DF contains the columns that verify whether the card id is valid or not
    """
    if data.empty:
        return data

    # * Check for valid personal card id
    data["is_personal_id"] = data[f"clean_{card_col}"].apply(
        PersonalIDValidator.is_valid_card
    )

    if print_info:
        print(f"# PERSONAL ID FOUND: {data['is_personal_id'].sum()}")
        sep_display()

    # * Check for valid passport id
    data["is_passport"] = data[f"clean_{card_col}"].apply(
        PassportValidator.is_valid_card
    )

    if print_info:
        print(f"# PASSPORT FOUND: {data['is_passport'].sum()}")
        sep_display()

    # * Check for valid driver license id
    data["is_driver_license"] = data[f"clean_{card_col}"].apply(
        DriverLicenseValidator.is_valid_card
    )

    if print_info:
        print(f"# DRIVER LICENSE FOUND: {data['is_driver_license'].sum()}")
        sep_display()

    # * Make a general is_valid column to verify whether the card is generally valid

    data["is_valid"] = data.apply(
        lambda row: is_checker_valid(
            [
                row["is_personal_id"],
                row["is_passport"],
                row["is_driver_license"],
            ]
        ),
        axis=1,
    )

    return data


def process_verify_card(
    data: pd.DataFrame, card_col: str = "card_id", n_cores: int = 1
) -> pd.DataFrame:
    """
    Verify whether the card ids are valid or not with multi-core

    Parameters
    ----------
    card_df : pd.DataFrame
        The input DF containing card id
    card_col : str, optional
        The column contain card id, by default "card_id"
    n_cores : int, optional
        Number of cores to process, by default 1

    Returns
    -------
    pd.DataFrame
        The final DF contains the columns that verify whether the card id is valid or not
        * `is_valid`: Card is valid
        * `is_personal_id`: Card is personal id
        * `is_passport`: Card is passport
        * `is_driver_license`: Card is driver license
        * `gender`: Gender parsed
        * `year_of_birth`: YOB parsed
        * `city`: City parsed
    """
    orig_cols = data.columns
    card_data = data

    # ? CLEAN CARD ID
    # * Removing na values
    clean_data, na_data = extract_null_values(card_data, card_col)

    # * Basic cleaning card_id
    print(">>> Cleansing card id: ", end="")
    start_time = time()
    clean_data = parallelize_dataframe(
        clean_data, clean_card_data, n_cores=n_cores, card_col=card_col
    )
    clean_time = time() - start_time
    print(f"Time takes {int(clean_time)//60}m{int(clean_time)%60}s")
    sep_display()

    # ? VALIDATE CARD ID
    print(">>> Verifying card id: ", end="")
    start_time = time()
    validated_data = parallelize_dataframe(
        clean_data,
        verify_card,
        n_cores=n_cores,
        card_col=card_col,
        print_info=False,
    )
    verify_time = time() - start_time
    print(f"Time takes {int(verify_time)//60}m{int(verify_time)%60}s")
    sep_display()

    # ? PARSING INFO FROM CARD ID
    print(">>> Extracting card info: ", end="")
    start_time = time()
    parsed_data = parallelize_dataframe(
        validated_data,
        extract_card_info,
        n_cores=n_cores,
        card_col=f"clean_{card_col}",
    )
    extract_time = time() - start_time
    print(f"Time takes {int(extract_time)//60}m{int(extract_time)%60}s")
    sep_display()

    # ? CONCAT ALL SEP CARD IDS
    validator_cols = [
        "is_valid",
        "is_personal_id",
        "is_passport",
        "is_driver_license",
    ]
    new_cols = [
        f"clean_{card_col}",
        *validator_cols,
        "gender",
        "year_of_birth",
        "city",
    ]

    final_card_df = pd.concat([parsed_data, na_data], ignore_index=True)

    final_card_df[validator_cols] = final_card_df[validator_cols].fillna(False)

    final_card_df = final_card_df[[*orig_cols, *new_cols]]

    return final_card_df
