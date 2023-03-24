
import sys
import time

import pandas as pd
from datetime import datetime

from preprocessing_pgp.name.enrich_name import process_enrich
from preprocessing_pgp.name.gender.predict_gender import process_predict_gender
from preprocessing_pgp.name.split_name import NameProcess
from preprocessing_pgp.utils import parallelize_dataframe

sys.path.append('/bigdata/fdp/cdp/source/core_profile/preprocess/utils')
from dict_utils import load_hdfs_data
from const import (
    hdfs,
    PATH_DICT,
    RAW_NAME_COLS,
    OLD_DICT_RAW_PATH,
    OLD_DICT_CLEAN_PATH,
    VIETNAMESE_WORD_REGEX
)

RAW_NAM_DICT_COLS = [
    'name_raw', 'name_id', 'name',
    'last_name', 'middle_name', 'first_name',
    'pronoun', 'gender', 'customer_type',
    'is_full_name', 'is_valid',
    'update_date'
]
CLEAN_NAME_DICT_COLS = [
    'name', 'name_id'
]
name_process = NameProcess()

def load_name_data(
    date: str
) -> dict:
    name_df_dict = {}
    for df_name in PATH_DICT.keys():
        print(f"\t{df_name}")
        name_df_dict[df_name] = load_hdfs_data(
            PATH_DICT[df_name],
            date=date,
            cols=RAW_NAME_COLS[df_name]
        )

        if df_name == 'frt_credit_mirae_off':
            name_df_dict[df_name].fillna({
                'last_name': '',
                'middle_name': '',
                'first_name': ''
            }, inplace=True)
            name_df_dict[df_name]['full_name'] = (
                name_df_dict[df_name]['last_name'].str.strip()
                + ' ' + name_df_dict[df_name]['middle_name'].str.strip()
                + ' ' + name_df_dict[df_name]['first_name'].str.strip()
            ).str.strip()

            name_df_dict[df_name].drop(columns=[
                'last_name', 'middle_name', 'first_name'
            ], inplace=True)
            RAW_NAME_COLS[df_name].remove('last_name')
            RAW_NAME_COLS[df_name].remove('middle_name')
            RAW_NAME_COLS[df_name].remove('first_name')
            RAW_NAME_COLS[df_name].append('full_name')

    return name_df_dict


def sep_pronoun(
    df: pd.DataFrame,
    col: str
) -> pd.DataFrame:
    return df[col].apply(
        lambda name: name_process.CleanName(name)[1]
    )


def daily_enhance_name(
    date: str,
    n_cores: int = 1
):
    """
    New Enhance name for raw data from DWH

    Parameters
    ----------
    date : str
        The date for processing
    n_cores : int, optional
        Number of core to process, by default 1
    """
    print(">>> Load name data")
    name_df_dict = load_name_data(date=date)

    # * Load unique name from data
    print(">>> Collect unique name")
    name_raw_list = set()

    for df_name in name_df_dict.keys():
        for col in RAW_NAME_COLS[df_name]:
            print(f"\t{df_name} - {col}")
            name_raw_list.update(
                name_df_dict[df_name][
                    ~name_df_dict[df_name][col].str.strip().isin([''])
                    & name_df_dict[df_name][col].notna()
                ][col]
                .str.replace(r'\s+', ' ', regex=True)
                .str.strip()
                .str.title()
                .unique()
            )

    # * Load old dictionaries
    print(">>> Load old name dict")
    old_raw_name_dict = pd.read_parquet(
        f'{OLD_DICT_RAW_PATH}/name_dict.parquet',
        filesystem=hdfs
    )
    old_clean_name_dict = pd.read_parquet(
        f'{OLD_DICT_CLEAN_PATH}/name_dict.parquet',
        filesystem=hdfs
    )

    max_old_id = old_raw_name_dict['name_id'].max()

    old_raw_name_dict_map = dict(zip(
        old_raw_name_dict['name'],
        old_raw_name_dict['name_id']
    ))

    # * Select new names
    print(">>> Select new name")
    name_df = pd.DataFrame({
        'name_raw': list(name_raw_list - set(old_raw_name_dict['name_raw'].unique()))
    })

    if name_df.empty:
        print(">>> No new name to process")
        return
    print(f">>> Number of new names: {len(name_df)}")

    # * Preprocess new names
    print(">>> Preprocess new name")
    process_name_df = process_enrich(
        name_df,
        name_col='name_raw',
        n_cores=n_cores
    ).rename(columns={
        'final': 'name'
    })

    process_name_df = process_predict_gender(
        process_name_df,
        name_col='name',
        n_cores=n_cores
    ).rename(columns={
        'gender_predict': 'gender'
    })

    process_name_df['pronoun'] = parallelize_dataframe(
        process_name_df,
        sep_pronoun,
        n_cores=n_cores,
        col='name_raw'
    )

    process_name_df['is_full_name'] = False

    process_name_df.loc[
        process_name_df['first_name'].notna()
        & process_name_df['last_name'].notna(),
        'is_full_name'
    ] = True

    # * Check for valid name
    print(">>> Check for valid name")
    process_name_df['is_valid'] = False
    process_name_df.loc[
        ~process_name_df['name'].str.strip().isin([''])
        & process_name_df['name'].notna()
        & process_name_df['name'].str.contains(VIETNAMESE_WORD_REGEX, na=False, regex=True),
        'is_valid'
    ] = True

    process_name_df.loc[
        process_name_df['name'].isna(),
        'name'
    ] = process_name_df['name_raw']

    # * Generate name_id for new name
    print(">>> Generate name id")
    process_name_df['name_id'] = process_name_df['name'].map(
        old_raw_name_dict_map
    )
    map_name_df = process_name_df[
        process_name_df['name_id'].isna()
    ][['name']]\
        .drop_duplicates()\
        .sort_values('name', ignore_index=True)
    map_name_df.index = map_name_df.index + max_old_id + 1
    map_name_df['name_id'] = map_name_df.index

    name_id_map = dict(zip(
        map_name_df['name'],
        map_name_df['name_id']
    ))

    # * Create raw dictionary
    print(">>> Create new raw dictionary")
    process_name_df.loc[
        process_name_df['name_id'].isna(),
        'name_id'
    ] = process_name_df['name']\
        .map(name_id_map)\
        .fillna(-1)
    process_name_df['name_id'] = process_name_df['name_id'].astype(int)
    process_name_df = process_name_df\
        .sort_values('name_id', ignore_index=True)

    process_name_df['update_date'] = datetime.strptime(date, '%Y-%m-%d')
    process_name_df = process_name_df[RAW_NAM_DICT_COLS]

    # * Create clean dictionary
    print(">>> Create new clean dictionary")
    final_clean_name_dict = (
        process_name_df[
            process_name_df['is_valid']
        ][CLEAN_NAME_DICT_COLS]
        .sort_values('name_id', ignore_index=True)
    )

    # * OUT DATA
    print(">>> Save dictionary data")
    pd.concat([old_raw_name_dict, process_name_df])\
        .sort_values('name_id', ignore_index=True)\
        .to_parquet(
            f'{OLD_DICT_RAW_PATH}/name_dict.parquet',
            filesystem=hdfs,
            index=False
        )

    pd.concat([old_clean_name_dict, final_clean_name_dict])\
        .drop_duplicates(ignore_index=True)\
        .to_parquet(
            f'{OLD_DICT_CLEAN_PATH}/name_dict.parquet',
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

    daily_enhance_name(TODAY, n_cores=20)

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
