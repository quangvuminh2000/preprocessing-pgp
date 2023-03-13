
"""
Module contains pipeline for best profile
"""

import sys
import multiprocessing as mp
from difflib import SequenceMatcher
from datetime import datetime, timedelta

import pandas as pd
from tqdm import tqdm
from unidecode import unidecode
from preprocessing_pgp.name.split_name import NameProcess
from preprocessing_pgp.utils import parallelize_dataframe

sys.path.append('/bigdata/fdp/cdp/source/core_profile/preprocess/utils')
from filter_profile import get_difference_data
from const import (
    hdfs,
    PREPROCESS_PATH,
    UTILS_PATH
)


def generate_similarity(
    data,
    name_src_col,
    name_target_col
):
    return data.apply(
        lambda col: SequenceMatcher(None, col[name_src_col], col[name_target_col]).ratio(),
        axis=1
    )


def LoadNameGender(
    date_str: str,
    key: str = "phone"
):
    # create
    cttvs = [
        "fo_vne",
        "ftel_fplay",
        "ftel_internet",
        "sendo_sendo",
        "frt_fshop",
        "frt_longchau",
        "fsoft_vio"
    ]
    if key == "phone":  # Credit data not having email
        cttvs = cttvs + ["frt_credit"]
    name_gender = pd.DataFrame()

    yesterday_str =\
        (datetime.strptime(date_str, '%Y-%m-%d') - timedelta(days=1))\
        .strftime('%Y-%m-%d')
    # loop
    # yesterday_str = '2023-02-20'
    for name_cttv in tqdm(cttvs):
        # read data
        data_cttv_today = pd.read_parquet(
            f"{PREPROCESS_PATH}/{name_cttv}.parquet/d={date_str}",
            filesystem=hdfs,
            columns=[key, "name", "customer_type", "gender"],
        ).dropna(subset=[key, 'name'], how='any')
        try:
            data_cttv_yesterday = pd.read_parquet(
                f"{PREPROCESS_PATH}/{name_cttv}.parquet/d={yesterday_str}",
                filesystem=hdfs,
                columns=[key, "name", "customer_type", "gender"],
            ).dropna(subset=[key, 'name'], how='any')

            # filters
            difference_data = get_difference_data(
                data_cttv_today, data_cttv_yesterday)
            difference_emails = difference_data[key].unique()

            data_cttv = data_cttv_today[data_cttv_today[key].isin(
                difference_emails)]
        except:
            data_cttv = data_cttv_today
        data_cttv["source_name"] = name_cttv.upper()

        # append
        name_gender = pd.concat([name_gender, data_cttv], ignore_index=True)

    # drop duplicate
    name_gender = name_gender.drop_duplicates(
        subset=[key, "name"], keep="first"
    )

    # return
    return name_gender


def CoreBestName(
    raw_names_n: pd.DataFrame,
    key: str = "phone",
    n_cores: int = 1
):
    # Skip name (non personal)
    print(">>> Skipping non-customer names")
    map_name_customer = raw_names_n.drop_duplicates(subset='raw_name')
    map_name_customer["num_word"] = map_name_customer["raw_name"].str.split(
        " ").str.len()
    skip_names = map_name_customer[
        (map_name_customer["customer_type"] != 'Ca nhan')
        | (map_name_customer["num_word"] > 4)
    ]["raw_name"].unique()

    skip_names_df = (
        raw_names_n[raw_names_n["raw_name"].isin(skip_names)]
        [[key, "raw_name"]]
        .drop_duplicates()
    )
    names_df = (
        raw_names_n[~raw_names_n["raw_name"].isin(skip_names)]
        [[key, "raw_name"]]
        .drop_duplicates()
    )

    print(">> Skip & Decode name components")
    # Split name: last, middle, first
    map_split_name = names_df[["raw_name"]].drop_duplicates()
    name_process = NameProcess()
    with mp.Pool(n_cores) as pool:
        map_split_name[["last_name", "middle_name", "first_name"]] =\
            pool.map(
                name_process.SplitName, map_split_name["raw_name"]
        )

    # unidecode first, last name
    first_name_exist_mask = map_split_name['first_name'].notna()
    last_name_exist_mask = map_split_name['last_name'].notna()
    map_split_name.loc[
        first_name_exist_mask,
        'unidecode_first_name'
    ] = map_split_name.loc[
        first_name_exist_mask,
        'first_name'
    ].apply(unidecode)
    map_split_name.loc[
        last_name_exist_mask,
        'unidecode_last_name'
    ] = map_split_name.loc[
        last_name_exist_mask,
        'last_name'
    ].apply(unidecode)
    names_df = pd.merge(
        names_df.set_index('raw_name'),
        map_split_name.set_index('raw_name'),
        left_index=True, right_index=True,
        how='left',
        sort=True
    ).reset_index()

    # * Create group_id
    print(">> Create group id based on first name & key")

    names_df["group_id"] = names_df[key] + \
        "-" + names_df["unidecode_first_name"]

    # Split case process best_name
    names_df["num_last_name"] = names_df.groupby(by=["group_id"])[
        "unidecode_last_name"
    ].transform("nunique")

    names_df = names_df.drop(columns=[
        "unidecode_first_name",
        "unidecode_last_name"
    ])

    info_name_columns = [
        "group_id",
        "raw_name",
        "last_name",
        "middle_name",
        "first_name",
    ]
    multi_last_name_mask = names_df['num_last_name'] >= 2
    multi_last_names_df =\
        names_df[multi_last_name_mask][info_name_columns]
    single_last_name_df =\
        names_df[~multi_last_name_mask][info_name_columns]

    # * Process case: 1 first_name - n last_name
    print(">> Processing multi-key last name")

    post_names_n_df = multi_last_names_df[
        multi_last_names_df["last_name"].isna()
    ]
    map_names_n_df = multi_last_names_df[
        multi_last_names_df["last_name"].notna()
        & multi_last_names_df["group_id"].isin(post_names_n_df["group_id"])
    ]

    # Generate sorting component
    map_names_n_df["num_char"] = map_names_n_df["raw_name"]\
        .str.len()
    map_names_n_df["num_word"] = map_names_n_df["raw_name"]\
        .str.split(" ").str.len()
    map_names_n_df["accented"] =\
        (map_names_n_df["raw_name"]
         != map_names_n_df["raw_name"].apply(unidecode))
    map_names_n_df = map_names_n_df.sort_values(
        by=["group_id", "num_word", "num_char", "accented"], ascending=False
    )
    map_names_n_df = map_names_n_df.groupby(by=["group_id"]).head(1)
    map_names_n_df = map_names_n_df[["group_id", "raw_name"]].rename(
        columns={"raw_name": "best_name"}
    )

    post_names_n_df = pd.merge(
        post_names_n_df.set_index('group_id'),
        map_names_n_df.set_index('group_id'),
        left_index=True, right_index=True,
        how='left',
        sort=True
    ).reset_index()

    post_names_n_df = post_names_n_df[["group_id", "raw_name", "best_name"]]

    multi_last_names_df = pd.merge(
        multi_last_names_df.set_index(['group_id', 'raw_name']),
        post_names_n_df.set_index(['group_id', 'raw_name']),
        left_index=True, right_index=True,
        how='left',
        sort=True
    ).reset_index()

    multi_last_names_df.loc[multi_last_names_df["best_name"].isna(
    ), "best_name"] = multi_last_names_df["raw_name"]
    multi_last_names_df = multi_last_names_df[[
        "group_id", "raw_name", "best_name"]]

    # * Process case: 1 first_name - 1 last_name
    print(">> Processing single-key last name")

    map_single_name_df = single_last_name_df[["group_id"]].drop_duplicates()

    for element_name in tqdm(["last_name", "middle_name", "first_name"]):
        # filter data detail
        map_element_name = (
            single_last_name_df[single_last_name_df[element_name].notna()]
            [["group_id", element_name]]
            .drop_duplicates()
        )

        # create features
        map_element_name[f"unidecode_{element_name}"] = map_element_name[
            element_name
        ].apply(unidecode)
        map_element_name["num_overall"] = map_element_name.groupby(
            by=["group_id", f"unidecode_{element_name}"]
        )[element_name].transform("count")
        map_element_name = map_element_name.drop(
            columns=f"unidecode_{element_name}")

        map_element_name["num_char"] = map_element_name[element_name].str.len()
        map_element_name["num_word"] = (
            map_element_name[element_name].str.split(" ").str.len()
        )
        map_element_name["accented"] = map_element_name[
            element_name
        ] != map_element_name[element_name].apply(unidecode)

        # approach to choose best name
        map_element_name = map_element_name.sort_values(
            by=["group_id", "accented", "num_overall", "num_word", "num_char"],
            ascending=False,
        )
        map_element_name = map_element_name.groupby(by=["group_id"]).head(1)
        map_element_name = map_element_name[["group_id", element_name]]
        map_element_name.columns = ["group_id", f"best_{element_name}"]

        # merge
        map_single_name_df = pd.merge(
            map_single_name_df.set_index('group_id'),
            map_element_name.set_index('group_id'),
            left_index=True, right_index=True,
            how='left',
            sort=True
        ).reset_index()

    # combine element name
    dict_trash = {
        "": None,
        "Nan": None,
        "nan": None,
        "None": None,
        "none": None,
        "Null": None,
        "null": None,
        "''": None,
    }
    columns = ["best_last_name", "best_middle_name", "best_first_name"]
    map_single_name_df["best_name"] = (
        map_single_name_df[columns]
        .fillna("")
        .agg(" ".join, axis=1)
        .str.replace("(?<![a-zA-Z0-9]),", "", regex=True)
        .str.replace("-(?![a-zA-Z0-9])", "", regex=True)
        .str.strip()
        .replace(dict_trash)
    )

    # merge
    single_last_name_df = pd.merge(
        single_last_name_df.set_index('group_id'),
        map_single_name_df[['group_id', 'best_name']].set_index('group_id'),
        left_index=True, right_index=True,
        how='left',
        sort=True
    ).reset_index()
    single_last_name_df = single_last_name_df[[
        "group_id", "raw_name", "best_name"]]

    # Concat
    print(">>> Concat to final name df")
    names_df = pd.concat([
        single_last_name_df,
        multi_last_names_df
    ], ignore_index=True)

    # Calculate similarity_score
    print(">>> Calculating similarity score")
    names_df['similarity_score'] =\
    parallelize_dataframe(
        names_df,
        generate_similarity,
        n_cores=n_cores,
        name_src_col='raw_name',
        name_target_col='best_name'
    )

    # Postprocess
    print(">> Post-processing")
    pre_names_df = names_df[
        ["group_id", "raw_name", "best_name", "similarity_score"]
    ]
    pre_names_df[key] = pre_names_df["group_id"].str.split("-").str[0]

    pre_names_df = pre_names_df.drop(columns=["group_id"])

    pre_names_df = pd.concat([
        pre_names_df,
        skip_names_df
    ], ignore_index=True)

    pre_names_df.loc[
        pre_names_df["best_name"].isna(),
        "best_name"
    ] = pre_names_df[
        "raw_name"
    ]

    pre_names_df.loc[
        pre_names_df["similarity_score"].isna(),
        "similarity_score"
    ] = 1

    # Merging with raw names
    print(">>> Merging back to raw names")
    pre_names_n = pd.merge(
        raw_names_n.set_index([key, 'raw_name']),
        pre_names_df.set_index([key, 'raw_name']),
        left_index=True, right_index=True,
        how='left',
        sort=True
    ).reset_index()
    pre_names_n.loc[
        pre_names_n["best_name"].isna(),
        "best_name"
    ] = pre_names_n[
        "raw_name"
    ]
    pre_names_n.loc[
        pre_names_n["similarity_score"].isna(),
        "similarity_score"
    ] = 1

    # Find source best name
    print(">>> Finding source best name")
    map_source_best_name = (
        pre_names_n.sort_values(
            by=[key, "best_name", "similarity_score"], ascending=False)
        .groupby(by=[key, "best_name"])
        .head(1)[[key, "best_name", "source_name"]]
    )

    map_source_best_name = map_source_best_name.rename(
        columns={"source_name": "source_best_name"}
    )

    pre_names_n = pd.merge(
        pre_names_n.set_index([key, 'best_name']),
        map_source_best_name.set_index([key, 'best_name']),
        left_index=True, right_index=True,
        how='left',
        sort=True
    ).reset_index()

    # Return
    return pre_names_n


def UniqueBestName(
    name_gender_by_key: pd.DataFrame,
    key: str,
) -> pd.DataFrame:
    # * Choosing unique name by priority
    print(">>> Setting priority sorting")
    priority_names = {
        "FTEL": 1,
        "SENDO": 2,
        "FPLAY": 3,
        "FO": 4,
        "FSHOP": 5,
        "LONGCHAU": 6,
        "FSOFT": 7
    }
    if key == 'phone':
        priority_names['CREDIT'] = 8

    name_gender_by_key['priority'] = name_gender_by_key['source_best_name'].map(priority_names)
    name_gender_by_key['is_customer'] =\
        name_gender_by_key['customer_type'] == 'Ca nhan'
    name_gender_by_key['is_good_length'] =\
        name_gender_by_key['best_name'].str.split(" ").str.len() <= 4

    stats_best_name =\
    name_gender_by_key.groupby(by=[key, 'best_name'])['best_name']\
    .agg(num_overall='count')\
    .reset_index()

    name_gender_by_key = pd.merge(
        name_gender_by_key.set_index([key, 'best_name']),
        stats_best_name.set_index([key, 'best_name']),
        left_index=True, right_index=True,
        how='left',
        sort=False
    ).reset_index()

    name_gender_by_key=\
    name_gender_by_key.sort_values(
        by=[key, 'is_customer', 'is_good_length', 'num_overall', 'priority'],
        ascending=[True, False, False, False, True]
    )

    name_gender_by_key = name_gender_by_key.drop(
        columns=['priority', 'num_overall', 'is_customer', 'is_good_length']
    )

    # * Generate unique name by key
    unique_name_gender_by_key =\
    name_gender_by_key.groupby(by=[key])\
    .head(1)[[key, 'best_name', 'best_gender', 'source_best_name']]

    unique_name_gender_by_key=\
    unique_name_gender_by_key.rename(columns={
        'best_name': 'unique_name',
        'best_gender': 'unique_gender',
        'source_best_name': 'source_unique_name'
    })

    name_gender_by_key = pd.merge(
        name_gender_by_key.set_index(key),
        unique_name_gender_by_key.set_index(key),
        left_index=True, right_index=True,
        how='left',
        sort=False
    ).reset_index()

    return name_gender_by_key


def PipelineBestName(
    date_str: str,
    key: str = "phone",
    n_cores: int = 1
):

    # load data
    raw_names = LoadNameGender(
        date_str,
        key
    )
    raw_names.rename(
        columns={
            'name': 'raw_name'
        },
        inplace=True
    )

    # split data to choose best name
    dup_keys = raw_names[raw_names[key].duplicated()][key].unique()
    multi_key_names = raw_names[raw_names[key].isin(dup_keys)]
    single_key_names = raw_names[~raw_names[key].isin(dup_keys)]

    # run pipeline best_name
    print(">>> Process multi-key name")
    pre_multi_key_names = CoreBestName(multi_key_names, key, n_cores=n_cores)

    # fake best_name
    print(">>> Process single-key name")
    pre_single_key_names = single_key_names
    pre_single_key_names["best_name"] = pre_single_key_names["raw_name"]
    pre_single_key_names["similarity_score"] = 1
    pre_single_key_names["source_best_name"] = pre_single_key_names["source_name"]

    # concat
    pre_names = pd.concat(
        [pre_single_key_names, pre_multi_key_names],
        ignore_index=True
    )

    # best gender
    print(">>> Merging with dictionary to find gender")
    pre_names_unique = pre_names[pre_names['gender'].isna()]['raw_name'].unique()
    dict_name_lst = pd.read_parquet(
        f'{UTILS_PATH}/dict_name_latest.parquet',
        filters=[('enrich_name', 'in', pre_names_unique)],
        filesystem=hdfs,
        columns=['enrich_name', 'gender']
    ).drop_duplicates(subset=['enrich_name'], keep='last')\
    .rename(columns={
        'gender': 'gender_dict'
    })

    pre_names = pd.merge(
        pre_names.set_index('raw_name'),
        dict_name_lst.set_index('enrich_name'),
        left_index=True, right_index=True,
        how='left',
        sort=False
    ).reset_index()\
    .rename(columns={
        'index': 'raw_name'
    })

    pre_names.loc[
        pre_names['gender'].isna(),
        'gender'
    ] = pre_names['gender_dict']

    pre_names = pre_names.drop(columns=['gender_dict'])

    print(">>> Finding best gender")
    best_name_gender = pre_names[
        (pre_names['best_name'].notna())
        & (pre_names['best_name'] == pre_names['raw_name'])
    ][['best_name', 'gender']].rename(columns={
        'gender': 'best_gender'
    }).drop_duplicates()\
    .dropna(subset=['best_gender'])\
    .drop_duplicates(subset=['best_name'], keep='first')

    pre_names = pd.merge(
        pre_names.set_index('best_name'),
        best_name_gender.set_index('best_name'),
        left_index=True, right_index=True,
        how='left', sort=False
    ).reset_index()
    pre_names.loc[
        pre_names["best_gender"].isna(),
        "best_gender"
    ] = None

    pre_names = pre_names.rename(
        columns={
            'raw_name': 'name'
        }
    )

    # UNIQUE BEST NAME
    print(">>> Finding unique best name for key")
    pre_names = UniqueBestName(
        pre_names,
        key=key
    )

    # Load & concat to new data
    try:
        old_best_name = pd.read_parquet(
            f'{UTILS_PATH}/name_by_{key}_latest.parquet',
            filesystem=hdfs
        )
        pre_names = pd.concat([
            pre_names,
            old_best_name
        ], ignore_index=True)\
            .sort_values(by=['best_gender'], ascending=False)\
            .drop_duplicates(subset=[key, 'name'], keep='first')
    except:
        print("No older data found >>> Keep this one as the origin")

    # return
    return pre_names
