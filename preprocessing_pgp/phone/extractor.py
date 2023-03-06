"""
Module to convert old type phones to their new type
"""
from time import time

import pandas as pd
from tqdm import tqdm
from halo import Halo

from preprocessing_pgp.phone.const import (
    SUB_PHONE_10NUM,
    SUB_PHONE_11NUM,
    SUB_TELEPHONE_10NUM,
    SUB_TELEPHONE_11NUM,
)
from preprocessing_pgp.phone.utils import basic_phone_preprocess
from preprocessing_pgp.utils import (
    parallelize_dataframe,
    sep_display
)
from preprocessing_pgp.phone.converter import (
    convert_mobi_phone,
    convert_phone_region
)
from preprocessing_pgp.phone.detector import (
    detect_mobi_phone_vendor,
    detect_tele_phone_vendor,
    detect_meaningful_phone
)

# ? ENVIRONMENT SETUP
tqdm.pandas()


# ? CHECK & EXTRACT FOR VALID PHONE
@Halo(
    text='Checking & Converting Valid Phone',
    color='cyan',
    spinner='dots7',
    text_color='magenta'
)
def extract_valid_phone(
    phones: pd.DataFrame,
    phone_col: str = "phone",
    print_info: bool = True
) -> pd.DataFrame:
    """
    Check for valid phone by pattern of head-code and convert the valid-old-code to new-code phone

    Parameters
    ----------
    phones : pd.DataFrame
        The DataFrame contains the phones
    phone_col : str, optional
        The columns which direct to the phones, by default "phone"
    print_info : bool, optional
        Whether to print the information of the run

    Returns
    -------
    pd.DataFrame
        The DataFrame with converted phone column and check if valid or not
    """
    # * Split na phone
    na_phones = phones[phones[phone_col].isna()]
    #! Prevent override the origin DF
    f_phones = phones[phones[phone_col].notna()]
    origin_cols = f_phones.columns

    # ? Preprocess phone with basic phone string clean up
    f_phones["clean_phone"] = f_phones[phone_col].apply(
        basic_phone_preprocess)

    print(f_phones)

    if print_info:
        print(
            f"# OF PHONE CLEAN : {f_phones.query(f'clean_phone != {phone_col}').shape[0]}",
            end="\n\n",
        )

        # print("Sample of non-clean phones:")
        # print(f_phones.query(f"clean_phone != {phone_col}"), end="\n\n\n")

    # ? Calculate the phone length for further preprocessing
    f_phones["phone_length"] = f_phones["clean_phone"].map(
        lambda x: len(str(x))
    )

    # ? Phone length validation: currently support phone number with length of 10 and 11.
    # ? Also, phone prefix has to be in the sub-phone dictionary.

    # * Length 10 - New
    mask_valid_new_sub_phone = (f_phones["phone_length"] == 10) & (
        f_phones["clean_phone"].str[:3].isin(SUB_PHONE_10NUM)
    )
    f_phones.loc[
        mask_valid_new_sub_phone,
        ["is_phone_valid", "is_mobi", "is_new_mobi"],
    ] = True

    if print_info:
        print(
            f"# OF MOBI PHONE 10 NUM VALID : {mask_valid_new_sub_phone.sum()}",
            end="\n\n\n",
        )

    # * Length 11 - Old
    mask_valid_old_sub_phone = (f_phones["phone_length"] == 11) & (
        f_phones["clean_phone"].str[:4].isin(SUB_PHONE_11NUM)
    )
    f_phones.loc[
        mask_valid_old_sub_phone,
        ["is_phone_valid", "is_mobi", "is_old_mobi"],
    ] = True

    if print_info:
        print(
            f"# OF MOBI PHONE 11 NUM VALID : {mask_valid_old_sub_phone.sum()}",
            end="\n\n\n",
        )

    # ? Correct phone numbers with old phone number format.
    mask_old_phone_format = f_phones["is_old_mobi"] == True

    f_phones.loc[mask_old_phone_format, "phone_convert"] = f_phones.loc[
        mask_old_phone_format, "clean_phone"
    ].map(convert_mobi_phone)

    if print_info:
        print(
            f"# OF OLD MOBI PHONE CONVERTED : {f_phones['phone_convert'].notna().sum()}")

        # print("Sample of converted MOBI phone:", end="\n\n")
        # print(f_phones.loc[(mask_old_phone_format) &
        #                    (f_phones["phone_convert"].notna())])

    # ? Check for valid tele-phone (old/new)

    # * Length 11 - NEW
    mask_valid_new_tele_phone = (
        (f_phones["phone_length"] == 11)
        & (
            (f_phones["clean_phone"].str[:3].isin(SUB_TELEPHONE_11NUM))
            | (f_phones["clean_phone"].str[:4].isin(SUB_TELEPHONE_11NUM))
        )
        & (f_phones["is_mobi"].isna())
    )
    f_phones.loc[
        mask_valid_new_tele_phone,
        ["is_phone_valid", "is_new_landline"],
    ] = True

    # * Length 10 - OLD
    mask_valid_old_tele_phone = (
        (f_phones["phone_length"] == 10)
        & (
            (f_phones["clean_phone"].str[:3].isin(SUB_TELEPHONE_10NUM))
            | (f_phones["clean_phone"].str[:2].isin(SUB_TELEPHONE_10NUM))
            | (f_phones["clean_phone"].str[:4].isin(SUB_TELEPHONE_10NUM))
        )
        & (f_phones["is_mobi"].isna())
    )
    f_phones.loc[
        mask_valid_old_tele_phone,
        ["is_phone_valid", "is_old_landline"],
    ] = True

    # ? Convert head phone of region from old to new

    mask_old_region_phone = f_phones["is_old_landline"] == True

    if print_info:
        print(f"# OF OLD REGION PHONE : {mask_old_region_phone.sum()}")

    f_phones.loc[mask_old_region_phone, "phone_convert"] = f_phones.loc[
        mask_old_region_phone, "clean_phone"
    ].map(convert_phone_region)

    # if print_info:
    #     print("Sample of converted telephone by region:", end="\n\n")
    #     print(f_phones.loc[(mask_old_region_phone) &
    #                        (f_phones["phone_convert"].notna())])

    # ? Filling NaNs in indicator columns

    fill_cols = [
        "is_phone_valid",
        "is_mobi",
        "is_new_mobi",
        "is_old_mobi",
        "is_new_landline",
        "is_old_landline"
    ]
    f_phones[fill_cols] = f_phones[fill_cols].fillna(False)

    # ? Final preprocessing - Case not changing any head code
    f_phones = f_phones.drop("phone_length", axis=1)
    f_phones.loc[
        f_phones["is_phone_valid"] & f_phones["phone_convert"].isna(
        ), "phone_convert"
    ] = f_phones["clean_phone"]

    if print_info:
        print(
            f"# OF VALID PHONE : {f_phones[f_phones['is_phone_valid']].shape[0]}",
            end="\n\n",
        )
        print(
            f"# OF INVALID PHONE : {f_phones[~f_phones['is_phone_valid']].shape[0]}",
            end="\n\n",
        )

        # print("Sample of invalid phones:", end="\n\n")

    f_phones.drop(phone_col, axis=1, inplace=True)
    f_phones.rename(columns={"clean_phone": phone_col}, inplace=True)
    f_phones = f_phones[[*origin_cols, *fill_cols, 'phone_convert']]
    # if print_info:
    #     print(f_phones[~f_phones["is_phone_valid"]].head(10))

    final_phones = pd.concat([f_phones, na_phones])
    final_phones[fill_cols] = final_phones[fill_cols].fillna(False)

    # ? Add Vendor
    valid_mobi_phone_mask = (final_phones['is_phone_valid'] &
                             final_phones['is_mobi'])
    final_phones.loc[
        valid_mobi_phone_mask,
        'phone_vendor'
    ] = final_phones.loc[
        valid_mobi_phone_mask,
        'phone_convert'
    ].apply(detect_mobi_phone_vendor)

    valid_tele_phone_mask = (final_phones['is_phone_valid'] &
                             ~final_phones['is_mobi'])
    final_phones.loc[
        valid_tele_phone_mask,
        'phone_vendor'
    ] = final_phones.loc[
        valid_tele_phone_mask,
        'phone_convert'
    ].apply(detect_tele_phone_vendor)

    # ? Detect meaningful phone
    final_phones.loc[
        final_phones['is_phone_valid'],
        'tail_phone_type'
    ] = final_phones.loc[
        final_phones['is_phone_valid'],
        'phone_convert'
    ].apply(detect_meaningful_phone)

    # ? Add phone type
    final_phones.loc[
        final_phones['is_mobi'],
        'phone_type'
    ] = 'mobile phone'
    final_phones.loc[
        (~final_phones['is_mobi']) & (final_phones['is_phone_valid']),
        'phone_type'
    ] = 'landline'

    return final_phones


def process_convert_phone(
    data: pd.DataFrame,
    phone_col: str = 'phone',
    n_cores: int = 1
) -> pd.DataFrame:
    """
    Converting valid phone to new phone type

    Parameters
    ----------
    data : pd.DataFrame
        The data containing the phone numbers
    phone_col : str, optional
        The column name contains records of phones, by default 'phone'
    n_cores : int, optional
        The number of core to process, by default 1

    Returns
    -------
    pd.DataFrame
        The converted data with new columns:
        * `is_phone_valid`: indicator for valid phone
        * `phone_type`: the phone type (mobi or landline)
        * `phone_convert`: converted valid old phone type to new phone type
        * `phone_vendor`: the vendor type of the phone number
        * `tail_phone_type`: the tail phone number meanings
    """

    # * Select only phone column
    orig_cols = data.columns
    phone_data = data[[phone_col]]

    # * Validate and convert phone
    start_time = time()

    converted_data = parallelize_dataframe(
        phone_data,
        extract_valid_phone,
        n_cores=n_cores,
        phone_col=phone_col,
        print_info=False
    )

    convert_time = time()-start_time

    print(
        f"Converting phones takes {int(convert_time)//60}m{int(convert_time)%60}s")
    sep_display()

    # * Concat with original cols
    new_cols = [
        'is_phone_valid',
        'phone_type',
        'phone_convert',
        'phone_vendor',
        'tail_phone_type'
    ]
    converted_data = converted_data[new_cols]

    converted_data = pd.concat([
        data[orig_cols],
        converted_data
    ], axis=1)

    return converted_data
