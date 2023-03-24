
import sys
import time

import pandas as pd
from datetime import datetime

from preprocessing_pgp.email.info_extractor import process_extract_email_info

sys.path.append('/bigdata/fdp/cdp/source/core_profile/preprocess/utils')
from const import (
    hdfs,
    PATH_DICT,
    RAW_EMAIL_COLS,
    OLD_DICT_RAW_PATH,
    OLD_DICT_CLEAN_PATH,
    DOMAIN_DICT_PATH
)
from dict_utils import load_hdfs_data


RAW_EMAIL_DICT_COLS = [
    'email_raw', 'email_id', 'email', 'phone',
    'user_name', 'gender', 'year_of_birth',
    'city', 'customer_type', 'email_group',
    'user_name_iscertain', 'is_auto_email',
    'is_valid', 'update_date'
]
CLEAN_EMAIL_DICT_COLS = [
    'email', 'email_id'
]

def load_email_data(
    date: str
) -> dict:
    email_df_dict = {}
    for df_name in PATH_DICT.keys():
        print(f"\t{df_name}")
        email_df_dict[df_name] = load_hdfs_data(
            PATH_DICT[df_name],
            date=date,
            cols=RAW_EMAIL_COLS[df_name]
        )

    return email_df_dict


def daily_enhance_email(
    date: str,
    n_cores: int = 1
):
    """
    New Enhance email for raw data from DWH

    Parameters
    ----------
    date : str
        The date for processing
    n_cores : int, optional
        Number of core to process, by default 1
    """
    print(">>> Load email data")
    email_df_dict = load_email_data(date=date)

    # * Load unique email from data
    print(">>> Collect unique email")
    email_raw_list = set()

    for df_name in email_df_dict.keys():
        for col in RAW_EMAIL_COLS[df_name]:
            print(f"\t{df_name} - {col}")
            email_raw_list.update(
                email_df_dict[df_name][
                    ~email_df_dict[df_name][col].str.strip().isin([''])
                    & email_df_dict[df_name][col].notna()
                ][col]
                .str.replace(r'\s+', ' ', regex=True)
                .str.strip()
                .str.lower()
                .unique()
            )

    # * Load old dictionaries
    print(">>> Load old email dict")
    old_raw_email_dict = pd.read_parquet(
        f'{OLD_DICT_RAW_PATH}/email_dict.parquet',
        filesystem=hdfs
    )
    old_clean_email_dict = pd.read_parquet(
        f'{OLD_DICT_CLEAN_PATH}/email_dict.parquet',
        filesystem=hdfs
    )

    max_old_id = old_raw_email_dict['email_id'].max()

    old_raw_email_dict_map = dict(zip(
        old_raw_email_dict['email'],
        old_raw_email_dict['email_id']
    ))

    # * Select new emails
    print(">>> Select new email")
    email_df = pd.DataFrame({
        'email_raw': list(email_raw_list - set(old_raw_email_dict['email_raw'].unique()))
    })

    if email_df.empty:
        print(">>> No new email to process")
        return
    print(f">>> Number of new emails: {len(email_df)}")

    # * Preprocess new emails
    print(">>> Preprocess new email")
    domain_dict = pd.read_parquet(
        DOMAIN_DICT_PATH,
        filesystem=hdfs
    )
    process_email_df = process_extract_email_info(
        email_df,
        email_col='email_raw',
        n_cores=n_cores,
        domain_dict=domain_dict
    )\
        .reset_index(drop=True)

    process_email_df = process_email_df.rename(columns={
        'cleaned_email_raw': 'email',
        'is_email_valid': 'is_valid',
        'address_extracted': 'city',
        'enrich_name': 'user_name',
        'gender_extracted': 'gender',
        'yob_extracted': 'year_of_birth',
        'phone_extracted': 'phone',
        'email_domain': 'email_group',
        'is_autoemail': 'is_auto_email',
        'username_iscertain': 'user_name_iscertain'
    })

    process_email_df.loc[
        process_email_df['email'].str.strip().isin([''])
        | process_email_df['email'].isna(),
        'email'
    ] = process_email_df['email_raw']

    # * Generate email_id for new email
    print(">>> Generate email id")
    process_email_df['email_id'] = process_email_df['email'].map(
        old_raw_email_dict_map
    )
    map_email_df = process_email_df[
        process_email_df['email_id'].isna()
    ][['email']]\
        .drop_duplicates()\
        .sort_values('email', ignore_index=True)
    map_email_df.index = map_email_df.index + max_old_id + 1
    map_email_df['email_id'] = map_email_df.index

    email_id_map = dict(zip(
        map_email_df['email'],
        map_email_df['email_id']
    ))

    # * Create raw dictionary
    print(">>> Create new raw dictionary")
    process_email_df.loc[
        process_email_df['email_id'].isna(),
        'email_id'
    ] = process_email_df['email']\
        .map(email_id_map)\
        .fillna(-1)
    process_email_df['email_id'] = process_email_df['email_id'].astype(int)
    process_email_df = process_email_df\
        .sort_values('email_id', ignore_index=True)

    process_email_df['update_date'] = datetime.strptime(date, '%Y-%m-%d')
    process_email_df = process_email_df[RAW_EMAIL_DICT_COLS]

    # * Create clean dictionary
    print(">>> Create new clean dictionary")
    final_clean_email_dict = (
        process_email_df[
            process_email_df['is_valid']
        ][CLEAN_EMAIL_DICT_COLS]
        .sort_values('email_id', ignore_index=True)
    )

    # * OUT DATA
    print(">>> Save dictionary data")
    pd.concat([old_raw_email_dict, process_email_df])\
        .sort_values('email_id', ignore_index=True)\
        .to_parquet(
            f'{OLD_DICT_RAW_PATH}/email_dict.parquet',
            filesystem=hdfs,
            index=False
        )

    pd.concat([old_clean_email_dict, final_clean_email_dict])\
        .drop_duplicates(ignore_index=True)\
        .to_parquet(
            f'{OLD_DICT_CLEAN_PATH}/email_dict.parquet',
            filesystem=hdfs,
            index=False
        )


if __name__ == '__main__':
    TODAY = sys.argv[1]

    print('')
    print('<=========================================START=========================================>')
    # start
    start = time.time()
    print(f'> start')

    daily_enhance_email(TODAY, n_cores=20)

    # end
    end = time.time()
    # delta
    delta = end - start
    # print time elapsed
    if delta > 60:
        if delta > 3600:
            print(f'> time elapsed: {delta//3600} h {(delta//60)%60} m {delta - (delta//60)*60} s')
        else:
            print(f'> time elapsed: {(delta//60)%60} m {delta - (delta//60)*60} s')
    else:
        print(f'> time elapsed: {delta} s')
    print('<=========================================END=========================================>')
    print('')
