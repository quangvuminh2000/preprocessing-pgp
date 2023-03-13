"""
Module to enhance general profile information
"""

import sys

import pandas as pd

from preprocessing_pgp.name.type.extractor import process_extract_name_type

sys.path.append('/bigdata/fdp/cdp/source/core_profile/preprocess/utils')
from const import (
    hdfs,
    DICT_NAME_UTILS_PATH,
    DICT_EMAIL_UTILS_PATH,
    DICT_PHONE_UTILS_PATH
)
from preprocess_profile import (
    cleansing_profile_name,
)


def enhance_name_profile(
    profile: pd.DataFrame,
    name_col: str = 'name',
    n_cores: int = 1
) -> pd.DataFrame:
    dict_trash = {
        '': None, 'Nan': None, 'nan': None,
        'None': None, 'none': None, 'Null': None,
        'null': None, "''": None
    }

    # * Cleansing
    print(">>> Cleansing profile name")
    profile = cleansing_profile_name(
        profile,
        name_col=name_col,
        n_cores=n_cores
    )
    profile = profile.rename(columns={
        name_col: 'raw_name'
    })

    profile_names = profile['raw_name'].drop_duplicates().dropna()

    if profile_names.empty:
        print(">>> No names found refilling info")
        profile[name_col] = None
        profile['last_name'] = None
        profile['middle_name'] = None
        profile['first_name'] = None
        profile['gender_enrich'] = None
        profile['customer_type'] = None
        return profile

    print(">>> Loading name dictionary")
    dict_name_lst = pd.read_parquet(
        DICT_NAME_UTILS_PATH,
        filters=[('raw_name', 'in', profile_names)],
        filesystem=hdfs,
        columns=[
            'raw_name', 'enrich_name',
            'last_name', 'middle_name', 'first_name',
            'gender',
        ]
    ).rename(columns={
        'gender': 'gender_enrich'
    })

    print(">>> Merging with name dictionary")
    profile = pd.merge(
        profile.set_index('raw_name'),
        dict_name_lst.set_index('raw_name'),
        left_index=True, right_index=True,
        how='left',
        sort=False
    ).rename(columns={
        'enrich_name': name_col
    }).reset_index(drop=False)

    print(">>> Refilling edge case name")
    cant_predict_name_mask =\
        (profile[name_col].isna()) & (profile['raw_name'].notna())
    profile.loc[
        cant_predict_name_mask,
        name_col
    ] = profile.loc[
        cant_predict_name_mask,
        'raw_name'
    ]
    profile[name_col] = profile[name_col].replace(dict_trash)

    print(">>> Processing Customer Type")
    profile = process_extract_name_type(
        profile,
        name_col=name_col,
        n_cores=n_cores,
        logging_info=False
    )
    profile['customer_type'] =\
        profile['customer_type'].map({
            'customer': 'Ca nhan',
            'company': 'Cong ty',
            'medical': 'Benh vien - Phong kham',
            'edu': 'Giao duc',
            'biz': 'Ho kinh doanh'
        })

    return profile


def enhance_email_profile(
    profile: pd.DataFrame,
    email_col: str = 'email',
) -> pd.DataFrame:
    profile = profile.rename(columns={
        email_col: 'email_raw'
    })

    profile_emails = profile['email_raw'].drop_duplicates().dropna()

    if profile_emails.empty:
        print(">>> No emails found refilling info")
        profile[email_col] = None
        profile['is_email_valid'] = False
        return profile

    print(">>> Loading email dictionary")
    valid_email = pd.read_parquet(
        DICT_EMAIL_UTILS_PATH,
        filters=[('email_raw', 'in', profile_emails)],
        filesystem=hdfs,
        columns=['email_raw', 'email', 'is_email_valid']
    )

    print(">>> Merging with email dictionary")
    profile = pd.merge(
        profile.set_index('email_raw'),
        valid_email.set_index('email_raw'),
        left_index=True, right_index=True,
        how='left',
        sort=False
    ).reset_index(drop=False)

    return profile


def enhance_phone_profile(
    profile: pd.DataFrame,
    phone_col: str = 'phone',
) -> pd.DataFrame:
    profile = profile.rename(columns={
        phone_col: 'phone_raw'
    })

    profile_phones = profile['phone_raw'].drop_duplicates().dropna()

    if profile_phones.empty:
        print(">>> No phones found refilling info")
        profile[phone_col] = None
        profile['is_phone_valid'] = False
        return profile

    print(">>> Loading phone dictionary")
    valid_phone = pd.read_parquet(
        DICT_PHONE_UTILS_PATH,
        filters=[('phone_raw', 'in', profile_phones)],
        filesystem=hdfs,
        columns=['phone_raw', 'phone', 'is_phone_valid']
    )

    print(">>> Merging with phone dictionary")
    profile = pd.merge(
        profile.set_index('phone_raw'),
        valid_phone.set_index('phone_raw'),
        left_index=True, right_index=True,
        how='left',
        sort=False
    ).reset_index(drop=False)

    return profile


def enhance_common_profile(
    profile: pd.DataFrame,
    enhance_phone: bool = True,
    enhance_email: bool = True,
    enhance_name: bool = True,
    n_cores: int = 1
) -> pd.DataFrame:

    if enhance_name:
        profile = enhance_name_profile(
            profile,
            name_col='name',
            n_cores=n_cores
        )

    if enhance_email:
        profile = enhance_email_profile(
            profile,
            email_col='email',
        )

    if enhance_phone:
        profile = enhance_phone_profile(
            profile,
            phone_col='phone',
        )

    return profile
