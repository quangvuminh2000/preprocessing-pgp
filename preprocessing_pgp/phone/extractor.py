import multiprocessing as mp

import pandas as pd
import numpy as np
from unidecode import unidecode
from tqdm import tqdm

from preprocessing_pgp.phone.const import (
    SUB_PHONE_10NUM,
    SUB_PHONE_11NUM,
    SUB_TELEPHONE_10NUM,
    SUB_TELEPHONE_11NUM,
)
from preprocessing_pgp.phone.utils import basic_phone_preprocess
from preprocessing_pgp.phone.converter import convert_mobi_phone, convert_phone_region

# ? ENVIRONMENT SETUP
tqdm.pandas()

N_PROCESSES = 20


# ? CHECK & EXTRACT FOR VALID PHONE
def extract_valid_phone(phones: pd.DataFrame, phone_col: str = "phone") -> pd.DataFrame:
    """
    Check for valid phone by pattern of head-code and convert the valid-old-code to new-code phone

    Parameters
    ----------
    f_phones : pd.DataFrame
        The DataFrame contains the phones
    phone_col : str, optional
        The columns which direct to the phones, by default "phone"

    Returns
    -------
    pd.DataFrame
        The DataFrame with converted phone column and check if valid or not
    """
    #! Prevent override the origin DF
    f_phones = phones.copy(deep=True)

    # ? Preprocess phone with basic phone string clean up
    f_phones["clean_phone"] = f_phones[phone_col].progress_apply(basic_phone_preprocess)

    print(
        f"# OF PHONE CLEAN : {f_phones.query(f'clean_phone != {phone_col}').shape[0]}",
        end="\n\n",
    )

    print("Sample of non-clean phones:")
    print(f_phones.query(f"clean_phone != {phone_col}"), end="\n\n\n")

    # ? Calculate the phone length for further preprocessing
    f_phones["phone_length"] = f_phones["clean_phone"].progress_map(
        lambda x: len(str(x))
    )

    # ? Phone length validation: currently support phone number with length of 10 and 11.
    # ? Also, phone prefix has to be in the sub-phone dictionary.

    # * Length 10
    mask_valid_new_sub_phone = (f_phones["phone_length"] == 10) & (
        f_phones["clean_phone"].str[:3].isin(SUB_PHONE_10NUM)
    )
    f_phones.loc[
        mask_valid_new_sub_phone,
        "is_phone_valid",
    ] = True

    print(
        f"# OF PHONE 10 NUM VALID : {mask_valid_new_sub_phone.sum()}",
        end="\n\n\n",
    )

    # * Length 11
    mask_valid_old_sub_phone = (f_phones["phone_length"] == 11) & (
        f_phones["clean_phone"].str[:4].isin(SUB_PHONE_11NUM)
    )
    f_phones.loc[
        mask_valid_old_sub_phone,
        "is_phone_valid",
    ] = True
    print(
        f"# OF PHONE 11 NUM VALID : {mask_valid_old_sub_phone.sum()}",
        end="\n\n\n",
    )

    f_phones = f_phones.reset_index(drop=True)

    # ? Correct phone numbers with old phone number format.
    mask_old_phone_format = (f_phones["phone_length"] == 11) & (
        f_phones["is_phone_valid"] == True
    )

    f_phones.loc[mask_old_phone_format, "phone_convert",] = f_phones.loc[
        mask_old_phone_format,
        "clean_phone",
    ].progress_map(convert_mobi_phone)

    print(f"# OF OLD MOBI PHONE CONVERTED : {f_phones['phone_convert'].notna().sum()}")

    print("Sample of converted MOBI phone:", end="\n\n")
    print(f_phones.loc[(mask_old_phone_format) & (f_phones["phone_convert"].notna())])

    # ? Check for valid tele-phone (old/new)
    mask_valid_new_tele_phone = (f_phones["phone_length"] == 11) & (
        (f_phones["clean_phone"].str[:3].isin(SUB_TELEPHONE_11NUM))
        | (f_phones["clean_phone"].str[:4].isin(SUB_TELEPHONE_11NUM))
    )
    f_phones.loc[
        mask_valid_new_tele_phone,
        "is_phone_valid",
    ] = True

    mask_valid_old_tele_phone = (f_phones["phone_length"] == 10) & (
        (f_phones["clean_phone"].str[:3].isin(SUB_TELEPHONE_10NUM))
        | (f_phones["clean_phone"].str[:2].isin(SUB_TELEPHONE_10NUM))
        | (f_phones["clean_phone"].str[:4].isin(SUB_TELEPHONE_10NUM))
    )
    f_phones.loc[
        mask_valid_old_tele_phone,
        "is_phone_valid",
    ] = True

    # ? Convert head phone of region from old to new

    mask_old_region_phone = (
        (f_phones["is_phone_valid"].notna())
        & (f_phones["is_phone_valid"])
        & (
            (f_phones["clean_phone"].str[:3].isin(SUB_TELEPHONE_10NUM))
            | (f_phones["clean_phone"].str[:2].isin(SUB_TELEPHONE_10NUM))
            | (f_phones["clean_phone"].str[:4].isin(SUB_TELEPHONE_10NUM))
        )
    )

    print(f"# OF OLD REGION PHONE : {mask_old_region_phone.sum()}")

    f_phones.loc[mask_old_region_phone, "phone_convert"] = f_phones.loc[
        mask_old_region_phone, "clean_phone"
    ].progress_map(convert_phone_region)

    print("Sample of converted telephone by region:", end="\n\n")
    print(
        f_phones.loc[
            (mask_old_region_phone) & (f_phones["phone_convert"].notna())
        ]
    )

    # ? Final preprocessing - Case not changing any head code

    f_phones["is_phone_valid"] = f_phones["is_phone_valid"].fillna(False)
    f_phones = f_phones.drop("phone_length", axis=1)
    f_phones.loc[
        f_phones["is_phone_valid"] & f_phones["phone_convert"].isna(), "phone_convert"
    ] = f_phones["clean_phone"]

    print(
        f"# OF VALID PHONE : {f_phones[f_phones['is_phone_valid']].shape[0]}",
        end="\n\n",
    )
    print(
        f"# OF INVALID PHONE : {f_phones[~f_phones['is_phone_valid']].shape[0]}",
        end="\n\n",
    )

    print("Sample of invalid phones:", end="\n\n")

    f_phones.drop(phone_col, axis=1, inplace=True)
    f_phones.rename(columns={"clean_phone": phone_col})
    print(f_phones[~f_phones["is_phone_valid"]].head(10))

    return f_phones
