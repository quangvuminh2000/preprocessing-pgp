
import sys
import time

import pandas as pd
from datetime import datetime

from preprocessing_pgp.phone.extractor import process_convert_phone

sys.path.append('/bigdata/fdp/cdp/source/core_profile/preprocess/utils')
from dict_utils import load_hdfs_data
from const import (
    hdfs,
    PATH_DICT,
    RAW_PHONE_COLS,
    OLD_DICT_RAW_PATH,
    OLD_DICT_CLEAN_PATH
)

RAW_PHONE_DICT_COLS = [
    'phone_raw', 'phone_clean', 'phone_id', 'phone',
    'vendor', 'type', 'vip_category',
    'is_valid', 'update_date'
]
CLEAN_PHONE_DICT_COLS = [
    'phone', 'phone_id'
]


def load_phone_data(
    date: str
) -> dict:
    phone_df_dict = {}
    for df_name in PATH_DICT.keys():
        print(f"\t{df_name}")
        phone_df_dict[df_name] = load_hdfs_data(
            PATH_DICT[df_name],
            date=date,
            cols=RAW_PHONE_COLS[df_name]
        )

    return phone_df_dict


def daily_enhance_phone(
    date: str,
    n_cores: int = 1
):
    """
    New Enhance phone for raw data from DWH

    Parameters
    ----------
    date : str
        The date for processing
    n_cores : int, optional
        Number of core to process, by default 1
    """
    print(">>> Load phone data")
    phone_df_dict = load_phone_data(date=date)

    # * Load unique phone from data
    print(">>> Collect unique phone")
    phone_raw_list = set()

    for df_name in phone_df_dict.keys():
        for col in RAW_PHONE_COLS[df_name]:
            print(f"\t{df_name} - {col}")
            phone_raw_list.update(
                phone_df_dict[df_name][
                    ~phone_df_dict[df_name][col].str.strip().isin([''])
                    & phone_df_dict[df_name][col].notna()
                ][col]
                .str.replace(r'\s+', ' ', regex=True)
                .str.strip()
                .str.lower()
                .unique()
            )

    # * Load old dictionaries
    print(">>> Load old phone dict")
    old_raw_phone_dict = pd.read_parquet(
        f'{OLD_DICT_RAW_PATH}/phone_dict.parquet',
        filesystem=hdfs
    )
    old_clean_phone_dict = pd.read_parquet(
        f'{OLD_DICT_CLEAN_PATH}/phone_dict.parquet',
        filesystem=hdfs
    )

    max_old_id = old_raw_phone_dict['phone_id'].max()

    old_raw_phone_dict_map = dict(zip(
        old_raw_phone_dict['phone'],
        old_raw_phone_dict['phone_id']
    ))

    # * Select new phones
    print(">>> Select new phone")
    phone_df = pd.DataFrame({
        'phone_raw': list(phone_raw_list - set(old_raw_phone_dict['phone_raw'].unique()))
    })

    if phone_df.empty:
        print(">>> No new phone to process")
        return
    print(f">>> Number of new phones: {len(phone_df)}")

    # * Preprocess new phones
    print(">>> Preprocess new phone")
    process_phone_df = process_convert_phone(
        phone_df,
        phone_col='phone_raw',
        n_cores=n_cores
    )\
        .reset_index(drop=True)

    process_phone_df = process_phone_df.rename(columns={
        'is_phone_valid': 'is_valid',
        'phone_type': 'type',
        'phone_convert': 'phone',
        'phone_vendor': 'vendor',
        'tail_phone_type': 'vip_category'
    })

    process_phone_df.loc[
        process_phone_df['phone_clean'].str.strip().isin([''])
        | process_phone_df['phone_clean'].isna(),
        'phone_clean'
    ] = process_phone_df['phone_raw']
    process_phone_df.loc[
        ~process_phone_df['is_valid'],
        'phone'
    ] = process_phone_df['phone_clean']

    # * Generate phone_id for new phone
    print(">>> Generate phone id")
    process_phone_df['phone_id'] = process_phone_df['phone'].map(
        old_raw_phone_dict_map
    )
    map_phone_df = process_phone_df[
        process_phone_df['phone_id'].isna()
    ][['phone']]\
        .drop_duplicates()\
        .sort_values('phone', ignore_index=True)
    map_phone_df.index = map_phone_df.index + max_old_id + 1
    map_phone_df['phone_id'] = map_phone_df.index

    phone_id_map = dict(zip(
        map_phone_df['phone'],
        map_phone_df['phone_id']
    ))

    # * Create raw dictionary
    print(">>> Create new raw dictionary")
    process_phone_df.loc[
        process_phone_df['phone_id'].isna(),
        'phone_id'
    ] = process_phone_df['phone']\
        .map(phone_id_map)\
        .fillna(-1)
    process_phone_df['phone_id'] = process_phone_df['phone_id'].astype(int)
    process_phone_df = process_phone_df\
        .sort_values('phone_id', ignore_index=True)

    process_phone_df['update_date'] = datetime.strptime(date, '%Y-%m-%d')
    process_phone_df = process_phone_df[RAW_PHONE_DICT_COLS]

    # * Create clean dictionary
    print(">>> Create new clean dictionary")
    final_clean_phone_dict = (
        process_phone_df[
            process_phone_df['is_valid']
        ][CLEAN_PHONE_DICT_COLS]
        .sort_values('phone_id', ignore_index=True)
    )

    # * OUT DATA
    print(">>> Save dictionary data")
    pd.concat([old_raw_phone_dict, process_phone_df])\
        .sort_values('phone_id', ignore_index=True)\
        .to_parquet(
            f'{OLD_DICT_RAW_PATH}/phone_dict.parquet',
            filesystem=hdfs,
            index=False
        )

    pd.concat([old_clean_phone_dict, final_clean_phone_dict])\
        .drop_duplicates(ignore_index=True)\
        .to_parquet(
            f'{OLD_DICT_CLEAN_PATH}/phone_dict.parquet',
            filesystem=hdfs,
            index=False
        )


if __name__ == '__main__':
    TODAY = sys.argv[1]

    print('')
    print('<=========================================START=========================================>')
    # start
    start = time.time()
    print('> start')

    daily_enhance_phone(TODAY, n_cores=20)

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
