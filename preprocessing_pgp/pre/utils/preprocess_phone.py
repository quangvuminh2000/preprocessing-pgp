
import pandas as pd

from const import (
    hdfs,
    VALID_PHONE_DICT
)

from preprocessing_pgp.phone.extractor import process_convert_phone


def validate_convert_phone(
    data: pd.DataFrame,
    dict_path: str,
    phone_col: str = 'phone',
    n_cores: int = 1
) -> pd.DataFrame:

    orig_cols = data.columns

    # * Na names & filter out name col
    na_data = data[data[phone_col].isna()][[phone_col]]
    cleaned_data = data[data[phone_col].notna()][[phone_col]]

    # * Merge with latest dict
    valid_phone = VALID_PHONE_DICT[[
        'phone_raw', 'phone_convert', 'is_phone_valid']]

    cleaned_data = pd.merge(
        cleaned_data.set_index(phone_col),
        valid_phone.set_index('phone_raw'),
        how='left',
        right_index=True,
        left_index=True
    ).reset_index()

    # * Proceed new phones
    new_phones = cleaned_data[cleaned_data['is_phone_valid'].isna()]
    old_phones = cleaned_data[cleaned_data['is_phone_valid'].notna()]

    new_phones = process_convert_phone(
        new_phones,
        phone_col=phone_col,
        n_cores=n_cores
    )

    # * Update dict to latest
