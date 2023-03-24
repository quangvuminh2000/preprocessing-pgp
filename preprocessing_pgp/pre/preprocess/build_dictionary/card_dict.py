import sys
import time
from typing import List
from datetime import datetime

import pandas as pd
from tqdm import tqdm

from preprocessing_pgp.card.validation import process_verify_card

sys.path.append('/bigdata/fdp/cdp/source/core_profile/preprocess/utils')
from const import (
    hdfs
)

FRT_CREDIT_FE_CREDIT_PATH = '/data/fpt/ftel/cads/dep_solution/user/trinhlk2/core_profile/build_dict/frt_credit_fe_credit_raw.parquet'
FRT_CREDIT_HOME_CREDIT_PATH = '/data/fpt/ftel/cads/dep_solution/user/trinhlk2/core_profile/build_dict/frt_credit_home_credit_raw.parquet'
FRT_CREDIT_MIRAE_OFF_PATH = '/data/fpt/ftel/cads/dep_solution/user/trinhlk2/core_profile/build_dict/frt_credit_mirae_off_raw.parquet'
FRT_CREDIT_MIRAE_ONL_PATH = '/data/fpt/ftel/cads/dep_solution/user/trinhlk2/core_profile/build_dict/frt_credit_mirae_onl_raw.parquet'
FRT_CREDIT_FSHOP_FORM_PATH = '/data/fpt/ftel/cads/dep_solution/sa/cdp/devices/raw/fshop_registration_form_installment.parquet/d=2023-02-26'
FRT_CREDIT_FSHOP_CUSTOMER_PATH = '/data/fpt/ftel/cads/dep_solution/sa/cdp/devices/raw/fshop_customer_installment.parquet/d=2023-02-26'

OLD_DICT_RAW_PATH = '/data/fpt/ftel/cads/dep_solution/user/quangvm9/migrate/rst/raw'
OLD_DICT_CLEAN_PATH = '/data/fpt/ftel/cads/dep_solution/user/quangvm9/migrate/rst/clean'

CARD_PATH_DICT = {
    'frt_credit_fe_credit': FRT_CREDIT_FE_CREDIT_PATH,
    'frt_credit_home_credit': FRT_CREDIT_HOME_CREDIT_PATH,
    'frt_credit_mirae_onl': FRT_CREDIT_MIRAE_ONL_PATH,
    'frt_credit_mirae_off': FRT_CREDIT_MIRAE_OFF_PATH,
    'frt_credit_fshop_form': FRT_CREDIT_FSHOP_FORM_PATH,
    'frt_credit_fshop_customer': FRT_CREDIT_FSHOP_CUSTOMER_PATH
}
RAW_CARD_COLS = {
    'frt_credit_fe_credit': ['customer_card_id'],
    'frt_credit_home_credit': ['card_id'],
    'frt_credit_mirae_onl': ['card_id'],
    'frt_credit_mirae_off': ['card_id', 'old_card_id'],
    'frt_credit_fshop_form': ['card_id', 'receiver_card_id'],
    'frt_credit_fshop_customer': ['id_card']
}
RAW_CARD_DICT_COLS = [
    'card_raw', 'card',
    'card_id', 'type',
    'gender', 'year_of_birth', 'city',
    'is_valid', 'update_date'
]
CLEAN_CARD_DICT_COLS = [
    'card', 'card_id', 'type'
]


def load_hdfs_data(
    path: str,
    date: str,
    cols: List[str] = []
) -> pd.DataFrame:
    return pd.read_parquet(
        path,
        columns=cols,
        filesystem=hdfs
    )


def load_card_data(
    date: str
) -> dict:
    card_df_dict = {}
    for df_name in CARD_PATH_DICT.keys():
        print(f"\t{df_name}")
        card_df_dict[df_name] = load_hdfs_data(
            CARD_PATH_DICT[df_name],
            date=date,
            cols=RAW_CARD_COLS[df_name]
        )

    return card_df_dict


# ? MAIN FUNCTION
def daily_enhance_card(
    date: str,
    n_cores: int = 1
):
    print(">>> Load card data")
    card_df_dict = load_card_data(date=date)

    # * Load unique card from data
    print(">>> Collect unique card")
    card_raw_list = set()

    for df_name in card_df_dict.keys():
        for col in RAW_CARD_COLS[df_name]:
            print(f"\t{df_name} - {col}")
            card_raw_list.update(
                card_df_dict[df_name][
                    ~card_df_dict[df_name][col].str.strip().isin([''])
                    & card_df_dict[df_name][col].notna()
                ][col]
                .str.replace(r'\s+', ' ', regex=True)
                .str.strip()
                .str.upper()
                .unique()
            )

    # * Load old dictionaries
    print(">>> Load old card dict")
    old_raw_card_dict = pd.read_parquet(
        f'{OLD_DICT_RAW_PATH}/card_dict.parquet',
        filesystem=hdfs
    )
    old_clean_card_dict = pd.read_parquet(
        f'{OLD_DICT_CLEAN_PATH}/card_dict.parquet',
        filesystem=hdfs
    )

    max_old_id = old_raw_card_dict['card_id'].max()

    old_raw_card_dict_map = dict(
        zip(old_raw_card_dict['card'], old_raw_card_dict['card_id']))

    # * Select new cards
    print(">>> Select new card")
    card_df = pd.DataFrame({
        'card_raw': list(card_raw_list-set(old_raw_card_dict['card_raw'].unique()))
    })

    if card_df.empty:
        print(">>> No new card to process")
        return
    print(f">>> Number of new cards: {len(card_df)}")

    # * Preprocess new cards
    print(">>> Preprocess new card")
    process_card_df = process_verify_card(
        card_df,
        'card_raw',
        n_cores=n_cores
    )\
        .reset_index(drop=True)
    process_card_df = process_card_df.rename(columns={
        'clean_card_raw': 'card'
    })

    # * Add card type
    print(">>> Add card type")
    process_card_df['type'] = None
    process_card_df['card_len'] = process_card_df['card'].str.len()

    process_card_df.loc[
        process_card_df['is_personal_id']
        & process_card_df['card_len'].isin([8, 11]),
        'card'
    ] = '0' + process_card_df['card']

    process_card_df['card_len'] = process_card_df['card'].str.len()

    process_card_df.loc[
        process_card_df['type'].isna()
        & process_card_df['is_personal_id']
        & process_card_df['card_len'].isin([12]),
        'type'
    ] = 'CCCD'

    process_card_df.loc[
        process_card_df['type'].isna()
        & process_card_df['is_personal_id']
        & process_card_df['card_len'].isin([9]),
        'type'
    ] = 'CMND'

    process_card_df.loc[
        process_card_df['type'].isna()
        & process_card_df['is_passport'],
        'type'
    ] = 'PASSPORT'

    process_card_df.loc[
        process_card_df['type'].isna()
        & process_card_df['is_driver_license'],
        'type'
    ] = 'GPLX'

    # * Extra process edge case
    print(">>> Extra process card")
    process_card_df.loc[
        ~process_card_df['is_valid'],
        'card'
    ] = process_card_df['card_raw']
    process_card_df['card'] = process_card_df['card'].str.upper()

    # * Generate card_id for new card
    print(">>> Generate card id")
    process_card_df['card_id'] = process_card_df['card'].map(
        old_raw_card_dict_map)
    map_card_df = process_card_df[
        process_card_df['card_id'].isna()
    ][['card']]\
        .drop_duplicates()\
        .sort_values('card', ignore_index=True)
    map_card_df.index = map_card_df.index + max_old_id + 1
    map_card_df['card_id'] = map_card_df.index

    card_id_map = dict(zip(map_card_df['card'], map_card_df['card_id']))

    # * Create raw dictionary
    print(">>> Create new raw dictionary")
    process_card_df.loc[
        process_card_df['card_id'].isna(),
        'card_id'
    ] = process_card_df['card']\
        .map(card_id_map)\
        .fillna(-1)
    process_card_df['card_id'] = process_card_df['card_id'].astype(int)
    process_card_df = process_card_df\
        .sort_values('card_id', ignore_index=True)

    process_card_df['update_date'] = datetime.strptime(date, '%Y-%m-%d')
    process_card_df = process_card_df[RAW_CARD_DICT_COLS]

    # * Create clean dictionary
    print(">>> Create new clean dictionary")
    final_clean_card_dict = (
        process_card_df[
            process_card_df['is_valid']
        ][CLEAN_CARD_DICT_COLS]
        .sort_values('card_id', ignore_index=True)
    )

    # * OUT DATA
    print(">>> Save dictionary data")
    pd.concat([old_raw_card_dict, process_card_df])\
        .sort_values('card_id', ignore_index=True)\
        .to_parquet(
            f'{OLD_DICT_RAW_PATH}/card_dict.parquet',
            filesystem=hdfs,
            index=False
        )

    pd.concat([old_clean_card_dict, final_clean_card_dict])\
        .drop_duplicates(ignore_index=True)\
        .to_parquet(
            f'{OLD_DICT_CLEAN_PATH}/card_dict.parquet',
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

    daily_enhance_card(TODAY, n_cores=20)

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
