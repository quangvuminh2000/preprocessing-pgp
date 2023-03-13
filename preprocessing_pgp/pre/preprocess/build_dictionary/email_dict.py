
import sys

import pandas as pd
from tqdm import tqdm

from preprocessing_pgp.email.info_extractor import process_extract_email_info

sys.path.append('/bigdata/fdp/cdp/source/core_profile/preprocess/utils')
from const import (
    hdfs,
    CENTRALIZE_PATH,
    UTILS_PATH,
    PRODUCT_PATH,
)


def load_email(
    cttv: str,
    day: str
) -> pd.DataFrame:

    emails = pd.read_parquet(
        f'{CENTRALIZE_PATH}/{cttv}.parquet/d={day}',
        filesystem=hdfs,
        columns=['email']  # Might change depends on each cttv
    ).drop_duplicates().dropna()

    return emails


def load_email_bank(
    cttvs,
    day: str
):
    email_bank = pd.DataFrame()
    for cttv in tqdm(cttvs):
        cttv_email = load_email(cttv, day)
        email_bank = pd.concat([
            email_bank,
            cttv_email
        ], ignore_index=True)

    email_bank = email_bank[~email_bank.duplicated()]
    return email_bank


def filter_difference_email(
    email_bank: pd.DataFrame,
    latest_check_email: pd.DataFrame
):
    email_bank = email_bank[
        ~email_bank['email'].isin(latest_check_email['email_raw'])
    ]

    return email_bank


def enhance_email(
    new_email: pd.DataFrame,
    email_col: str = 'email_raw',
    n_cores: int = 1
) -> pd.DataFrame:
    if new_email.empty:
        return new_email

    new_enhance_email = process_extract_email_info(
        new_email,
        email_col=email_col,
        n_cores=n_cores
    )
    new_enhance_email = new_enhance_email.rename(
        columns={
            f'cleaned_{email_col}': 'email',
            email_col: 'email_raw',
            'email_domain': 'email_group',
            'yob_extracted': 'year_of_birth',
            'phone_extracted': 'phone',
            'address_extracted': 'address',
            'gender_extracted': 'gender',
            'enrich_name': 'username',
            f'{email_col}_name': 'email_name'
        }
    )

    return new_enhance_email


def update_email_dict(
    new_enhance_email: pd.DataFrame,
    latest_check_email: pd.DataFrame,
    day: str
):
    new_enhance_email['export_date'] = day

    # Update latest email
    latest_check_email = pd.concat([
        new_enhance_email,
        latest_check_email
    ], ignore_index=True)

    # Save to utils
    latest_check_email.to_parquet(
        f'{UTILS_PATH}/valid_email_latest.parquet',
        filesystem=hdfs,
        index=False
    )
    # Save to product -- only valid email
    latest_valid_email = latest_check_email[latest_check_email['is_email_valid']]
    email_cols = [
        'email', 'username_iscertain', 'username',
        'year_of_birth', 'phone', 'address', 'email_group',
        'is_autoemail', 'gender', 'customer_type'
    ]
    latest_valid_email[email_cols].to_parquet(
        f'{PRODUCT_PATH}/valid_email_latest.parquet',
        filesystem=hdfs,
        index=False
    )


# ? MAIN FUNCTION
def daily_enhance_email(
    day: str,
    n_cores: int = 1
):
    email_cttv = [
        "fo_vne",
        "ftel_fplay",
        "ftel_internet",
        "sendo_sendo",
        "frt_fshop",
        "frt_longchau",
        "fsoft_vio"
    ]

    print(">>> Loading email from CTTV")
    email_banks = load_email_bank(email_cttv, day)

    print(">>> Loading dictionary email")
    latest_check_emails = pd.read_parquet(
        f'{UTILS_PATH}/valid_email_latest.parquet', filesystem=hdfs)

    print(">>> Filtering new email")
    new_email = filter_difference_email(
        email_banks,
        latest_check_emails
    )

    print(f"Number of new profile: {new_email.shape[0]}")

    print(">>> Enhancing new email")
    new_enhance_email = enhance_email(
        new_email,
        email_col='email',
        n_cores=n_cores
    )

    print(">>> Updating email dictionary")
    update_email_dict(
        new_enhance_email,
        latest_check_emails,
        day
    )


if __name__ == '__main__':
    TODAY = sys.argv[1]

    daily_enhance_email(TODAY, n_cores=20)
