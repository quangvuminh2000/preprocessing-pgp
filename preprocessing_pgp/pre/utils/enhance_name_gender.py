import sys

import pandas as pd
from tqdm import tqdm

from preprocessing_pgp.name.enrich_name import process_enrich
from preprocessing_pgp.name.gender.predict_gender import process_predict_gender

from const import (
    hdfs,
    RAW_PATH,
    UTILS_PATH,
    PRODUCT_PATH
)


def load_name_info(
    cttv: str,
    day: str
) -> pd.DataFrame:

    name_info = pd.read_parquet(
        f'{RAW_PATH}/{cttv}.parquet/d={day}',
        filesystem=hdfs,
        columns=['name', 'gender']
    )

    name_info = name_info[
        (name_info['name'].notna())
        & (~name_info.duplicated(subset=['name'], keep='first'))
    ]

    return name_info


def load_name_info_bank(
    cttvs,
    day: str
) -> pd.DataFrame:
    name_info_bank = pd.DataFrame()
    for cttv in tqdm(cttvs):
        name_info_cttv = load_name_info(cttv, day)
        name_info_bank = pd.concat([
            name_info_bank,
            name_info_cttv
        ], ignore_index=True)

    name_info_bank = name_info_bank[
        ~name_info_bank.duplicated(subset=['name'], keep='first')
    ]

    return name_info_bank


def filter_difference_name_info(
    name_info_bank: pd.DataFrame,
    dict_name: pd.DataFrame,
    bank_name_col: str,
    dict_name_col: str
) -> pd.DataFrame:
    name_info_bank = name_info_bank[
        ~name_info_bank[bank_name_col].isin(dict_name[dict_name_col])
    ]

    return name_info_bank



def enhance_name_info(
    new_name_info: pd.DataFrame,
    name_col: str = 'name',
    gender_col: str = 'gender',
    n_cores: int = 1
) -> pd.DataFrame:

    # * Enrich names
    enhance_name = process_enrich(
        new_name_info,
        name_col=name_col,
        n_cores=n_cores
    )

    # * Predicting gender
    new_enhance_name_info = process_predict_gender(
        enhance_name,
        name_col='final',
        n_cores=n_cores
    )

    # * Masking confused gender
    confused_gender_mask =\
        (new_enhance_name_info[gender_col].notna())\
        & (new_enhance_name_info[gender_col] != new_enhance_name_info["gender_predict"])

    new_enhance_name_info[confused_gender_mask, "is_confused"] = True
    new_enhance_name_info["is_confused"] = new_enhance_name_info["is_confused"].fillna(False)

    # * Mask fullname
    fullname_mask =\
        (new_enhance_name_info["first_name"].notna())\
        & (new_enhance_name_info["last_name"].notna())
    new_enhance_name_info.loc[
        fullname_mask,
        "is_full_name"
    ] = True

    new_enhance_name_info["is_full_name"] =\
        new_enhance_name_info["is_full_name"].fillna(False)

    new_enhance_name_info =\
        new_enhance_name_info.rename(
            columns={
                "name": "raw_name",
                "final": "enrich_name",
                "gender": "gender_raw"
            },
            inplace=True,
        )

    return new_enhance_name_info


def update_name_info_dict(
    new_enhance_name_info: pd.DataFrame,
    dict_name: pd.DataFrame,
    day:str
):
    new_enhance_name_info['d'] = day

    # Update latest name info
    dict_name = pd.concat([
        new_enhance_name_info,
        dict_name
    ], ignore_index=True)

    # Save to backups
    dict_name.to_parquet(
        '/data/fpt/ftel/cads/dep_solution/user/quangvm9/data/enrich/dict_name_enrich_new.parquet',
        filesystem=hdfs,
        index=False
    )

    # Save to utils
    dict_name_cols = [
        "raw_name",
        "enrich_name",
        "last_name",
        "middle_name",
        "first_name",
        # "is_full_name",
        "customer_type",
        # "gender_raw",
        "gender_predict",
        # "is_confused",
        "d",
    ]
    dict_name = dict_name[dict_name_cols]
    dict_name.to_parquet(
        f'{UTILS_PATH}/dict_name_latest_new.parquet',
        filesystem=hdfs,
        index=False
    )

    # Save to product
    dict_name_product = dict_name.drop(
        columns=['d']
    )

    dict_name_product.to_parquet(
        f'{PRODUCT_PATH}/dict_name_latest_new.parquet',
        filesystem=hdfs,
        index=False
    )







