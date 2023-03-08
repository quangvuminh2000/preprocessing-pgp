import sys

import pandas as pd
from tqdm import tqdm

from preprocessing_pgp.phone.extractor import process_convert_phone

sys.path.append('/bigdata/fdp/cdp/source/core_profile/preprocess/utils')
from const import (
    hdfs,
    CENTRALIZE_PATH,
    UTILS_PATH,
    PRODUCT_PATH,
)


def load_phone(
    cttv: str,
    day: str
) -> pd.DataFrame:
    phones = pd.read_parquet(
        f'{CENTRALIZE_PATH}/{cttv}.parquet/d={day}',
        filesystem=hdfs,
        columns=['phone']  # Might change depends on each cttv
    ).drop_duplicates().dropna()

    return phones


def load_phone_bank(
    cttvs,
    day: str
):
    phone_bank = pd.DataFrame()
    for cttv in tqdm(cttvs):
        cttv_phone = load_phone(cttv, day)
        phone_bank = pd.concat([
            phone_bank,
            cttv_phone
        ], ignore_index=True)

    phone_bank = phone_bank[~phone_bank.duplicated()]
    return phone_bank


def filter_difference_phone(
    phone_bank: pd.DataFrame,
    latest_check_phone: pd.DataFrame
):
    phone_bank = phone_bank[
        ~phone_bank['phone'].isin(latest_check_phone['phone_raw'])
    ]

    return phone_bank


def enhance_phone(
    new_phone: pd.DataFrame,
    phone_col: str = 'phone_raw',
    n_cores: int = 1
) -> pd.DataFrame:
    if new_phone.empty:
        return new_phone

    new_enhance_phone = process_convert_phone(
        new_phone,
        phone_col=phone_col,
        n_cores=n_cores
    )
    new_enhance_phone = new_enhance_phone.rename(
        columns={
            f'phone_convert': 'phone',
            phone_col: 'phone_raw'
        }
    )

    return new_enhance_phone


def update_phone_dict(
    new_enhance_phone: pd.DataFrame,
    latest_check_phone: pd.DataFrame,
    day: str
):
    new_enhance_phone['export_date'] = day

    # Update latest email
    latest_check_phone = pd.concat([
        new_enhance_phone,
        latest_check_phone
    ], ignore_index=True)

    # Save to utils
    latest_check_phone.to_parquet(
        f'{UTILS_PATH}/valid_phone_latest.parquet',
        filesystem=hdfs,
        index=False
    )
    # Save to product -- only valid email
    latest_valid_phone = latest_check_phone[latest_check_phone['is_phone_valid']]
    email_cols = [
        'phone_raw', 'phone', 'phone_vendor',
        'phone_type', 'tail_phone_type'
    ]
    latest_valid_phone[email_cols].to_parquet(
        f'{PRODUCT_PATH}/valid_phone_latest.parquet',
        filesystem=hdfs,
        index=False
    )


# ? MAIN FUNCTION
def daily_enhance_phone(
    day: str,
    n_cores: int = 1
):
    phone_cttv = [
        "fo_vne",
        "ftel_fplay",
        "ftel_internet",
        "sendo_sendo",
        "frt_fshop",
        "frt_longchau",
        "fsoft_vio",
        'frt_credit'
    ]

    print(">>> Loading email from CTTV")
    phone_bank = load_phone_bank(phone_cttv, day)

    print(">>> Loading dictionary email")
    latest_check_phone = pd.read_parquet(
        f'{UTILS_PATH}/valid_phone_latest.parquet', filesystem=hdfs)

    print(">>> Filtering new email")
    new_phone = filter_difference_phone(
        phone_bank,
        latest_check_phone
    )

    n_new_profile = new_phone.shape[0]
    print(f"Number of new profile: {new_phone.shape[0]}")

    if n_new_profile != 0:
        print(">>> Enhancing new email")
        new_enhance_phone = enhance_phone(
            new_phone,
            phone_col='phone',
            n_cores=n_cores
        )

        print(">>> Updating email dictionary")
        update_phone_dict(
            new_enhance_phone,
            latest_check_phone,
            day
        )
    else:
        print(">>> No new profile to update")


if __name__ == '__main__':
    TODAY = sys.argv[1]

    daily_enhance_phone(TODAY, n_cores=20)
