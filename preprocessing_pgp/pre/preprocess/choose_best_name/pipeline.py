
"""
Module contains pipeline for best profile
"""

import sys
from difflib import SequenceMatcher
from datetime import datetime, timedelta

import pandas as pd
from unidecode import unidecode
from preprocessing_pgp.utils import parallelize_dataframe

sys.path.append('/bigdata/fdp/cdp/source/core_profile/preprocess/utils')
from filter_profile import get_difference_data
from const import (
    hdfs,
    PREPROCESS_PATH,
    UTILS_PATH,
    OLD_DICT_RAW_PATH
)


def generate_similarity(
    data,
    name_src_col,
    name_target_col
):
    return data.apply(
        lambda col: SequenceMatcher(
            None, col[name_src_col], col[name_target_col]).ratio(),
        axis=1
    )


def LoadNameByKey(
    date_str: str,
    key: str = "phone",
    init: bool = False
):
    # create
    cttvs = [
        "fo_vne",
        "ftel_fplay",
        "ftel_internet",
        "sendo_sendo",
        "frt_fshop",
        "frt_longchau",
        "fsoft_vio",
        "frt_credit"
    ]
    name_df = pd.DataFrame()

    yesterday_str =\
        (datetime.strptime(date_str, '%Y-%m-%d') - timedelta(days=1))\
        .strftime('%Y-%m-%d')
    # loop
    # yesterday_str = '2023-02-20'
    for name_service in cttvs:
        # read data
        data_cttv_today = pd.read_parquet(
            f"{PREPROCESS_PATH}/{name_service}.parquet/d={date_str}",
            filesystem=hdfs,
            columns=[key, "name", "gender", "customer_type"],
        ).dropna(subset=[key, 'name'], how='any')
        if not init:
            try:
                data_cttv_yesterday = pd.read_parquet(
                    f"{PREPROCESS_PATH}/{name_service}.parquet/d={yesterday_str}",
                    filesystem=hdfs,
                    columns=[key, "name", "gender", "customer_type"],
                ).dropna(subset=[key, 'name'], how='any')

                # filters
                difference_data = get_difference_data(
                    data_cttv_today,
                    data_cttv_yesterday
                )
                difference_keys = difference_data[key].unique()

                data_service = data_cttv_today[
                    data_cttv_today[key].isin(difference_keys)
                ]
            except FileNotFoundError:
                data_service = data_cttv_today
        else:
            data_service = data_cttv_today
        data_service["source_name"] = name_service.upper()

        # append
        name_df = pd.concat([name_df, data_service], ignore_index=True)
        print(
            f"\t>> {name_service} : {data_service.shape[0]} new name-{key} pairs")

    # return
    return name_df


def CoreBestName(
    raw_names_n: pd.DataFrame,
    key: str = "phone_id",
    n_cores: int = 1
):
    # Skip name (non personal)
    print(">>> Skipping non-customer & non-Vietnamese names")
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
        .drop_duplicates(subset=[key, "raw_name"])
    )
    names_df = (
        raw_names_n[~raw_names_n["raw_name"].isin(skip_names)]
        [[key, "raw_name", "last_name", "middle_name", "first_name"]]
        .drop_duplicates(subset=[key, "raw_name"])
    )

    print(">> Decode name components")
    # unidecode first, last name
    names_df.loc[
        names_df['first_name'].notna(),
        'unidecode_first_name'
    ] = names_df.loc[
        names_df['first_name'].notna(),
        'first_name'
    ].apply(unidecode)
    names_df.loc[
        names_df['last_name'].notna(),
        'unidecode_last_name'
    ] = names_df.loc[
        names_df['last_name'].notna(),
        'last_name'
    ].apply(unidecode)

    # * Create group_id
    print(">>> Create group id based on first name & key")

    names_df["group_id"] = names_df[key].astype(str) + \
        "-" + names_df["unidecode_first_name"]

    # * Split case process best_name
    print(">>> Splitting multi-lastname and single-lastname")
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
    print(">>> Processing multi-lastname for multi-key")

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
        by=["group_id", "num_word", "accented", "num_char"], ascending=False
    )  # number of word -> accented -> num_char
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
        "group_id", "raw_name", "best_name"
    ]]

    # * Process case: 1 first_name - 1 last_name
    print(">>> Processing single-lastname for multi-key")

    map_single_name_df = single_last_name_df[["group_id"]].drop_duplicates()

    print("\t>> Finding best name element for each key")
    for element_name in ["last_name", "middle_name", "first_name"]:
        print(f"\t\t> Best {element_name}")
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
            columns=f"unidecode_{element_name}"
        )

        map_element_name["num_char"] = map_element_name[element_name].str.len()
        map_element_name["num_word"] = (
            map_element_name[element_name].str.split(" ").str.len()
        )
        map_element_name["accented"] = map_element_name[
            element_name
        ] != map_element_name[element_name].apply(unidecode)

        # approach to choose best name
        if element_name == 'middle_name':
            map_element_name["good_middle_name"] =\
                (map_element_name[element_name].str.split(' ').str.len() <= 2)\
                & (map_element_name[element_name].str.split(' ').str.len() > 0)
            map_element_name = map_element_name.sort_values(
                by=[
                    "group_id", "accented", "num_overall",
                    "good_middle_name", "num_word", "num_char"
                ],
                ascending=False,
            )  # accented -> number of repetition -> the standard of middle name -> number of word -> number of character
        else:  # first & last
            map_element_name = map_element_name.sort_values(
                by=["group_id", "accented", "num_overall", "num_word", "num_char"],
                ascending=False,
            )  # accented -> number of repetition -> number of word -> number of character
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
        .str.replace(r'\s+', ' ', regex=True)
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
        "group_id", "raw_name", "best_name"
    ]]

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
    pre_names_df[key] = pre_names_df["group_id"].str.split("-").str[0].astype(int)

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

    # * Merging with raw names
    print(">>> Merge back to raw names")
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

    # * Find source best name
    print(">>> Find source best name")
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
    key: str = 'phone_id',
) -> pd.DataFrame:
    # * Choosing unique name by priority
    print(">>> Setting priority sorting")
    priority_names = {
        "FTEL_INTERNET": 1,
        "SENDO_SENDO": 2,
        "FRT_CREDIT": 3,
        "FTEL_FPLAY": 4,
        "FO_VNE": 5,
        "FRT_FSHOP": 6,
        "FRT_LONGCHAU": 7,
        "FSOFT_VIO": 8,
    }

    name_gender_by_key['priority'] = name_gender_by_key['source_best_name'].map(
        priority_names)
    name_gender_by_key['is_customer'] =\
        name_gender_by_key['customer_type'] == 'Ca nhan'
    name_gender_by_key['is_good_length'] =\
        name_gender_by_key['best_name'].str.split(" ").str.len() <= 4
    name_gender_by_key['num_word'] =\
        name_gender_by_key['best_name'].str.split(" ").str.len()

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

    name_gender_by_key =\
        name_gender_by_key.sort_values(
            by=[key, 'is_customer', 'is_good_length', 'num_overall', 'num_word', 'priority'],
            ascending=[True, False, False, False, False, True]
        )  # Is Personal Name -> Having good length -> Number of repetition -> Service priority

    name_gender_by_key = name_gender_by_key.drop(
        columns=['priority', 'num_overall', 'is_customer', 'is_good_length', 'num_word']
    )

    # * Generate unique name by key
    unique_name_gender_by_key =\
        name_gender_by_key.groupby(by=[key])\
        .head(1)[[
            key, 'best_name',
            'best_gender', 'best_customer_type',
            'source_best_name'
        ]]

    unique_name_gender_by_key =\
        unique_name_gender_by_key.rename(columns={
            'best_name': 'unique_name',
            'best_gender': 'unique_gender',
            'best_customer_type': 'unique_customer_type',
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
    n_cores: int = 1,
    init: bool = False
):
    key_id = f'{key}_id'

    # * load data
    print(">>> Load name data by key")
    raw_names = LoadNameByKey(
        date_str,
        key,
        init=init
    )
    raw_names.rename(
        columns={
            'name': 'raw_name'
        },
        inplace=True
    )
    print(f"Total name-{key} pairs: {raw_names.shape[0]}")

    # * Merge to get hash key data
    if key == 'email':
        raw_names[key] = raw_names[key]\
            .str.replace(r'\s+', ' ', regex=True)\
            .str.strip()\
            .str.lower()
    if key == 'phone':
        raw_names[key] = raw_names[key]\
            .str.replace(r'\s+', ' ', regex=True)\
            .str.strip()\
            .str.lower()

    unique_keys = raw_names[key].unique()

    key_dict_raw = pd.read_parquet(
        f'{OLD_DICT_RAW_PATH}/{key}_dict.parquet',
        filesystem=hdfs,
        columns=[f'{key}_raw', f'{key}_id'],
        filters=[(f'{key}_raw', 'in', unique_keys)]
    )
    raw_names = pd.merge(
        raw_names.set_index(key),
        key_dict_raw.set_index(f'{key}_raw'),
        left_index=True, right_index=True,
        how='left',
        sort=False
    ).reset_index(drop=True)
    raw_names = raw_names.dropna(subset=[key_id], how='any')
    raw_names[key_id] = raw_names[key_id].astype(int)

    # * Merge data from dictionary names
    print(">>> Merging with name_dict for name information")
    name_dict_raw = pd.read_parquet(
        f'{OLD_DICT_RAW_PATH}/name_dict.parquet',
        filesystem=hdfs,
        columns=[
            'name_raw', 'name', 'name_id',
            'last_name', 'middle_name', 'first_name',
            'gender', 'customer_type'
        ],
        filters=[('name_raw', 'in', raw_names['raw_name'].unique())]
    ).rename(columns={
        'name_raw': 'raw_name',
        'gender': 'gender_dict',
        'customer_type': 'customer_type_dict'
    })

    raw_names['raw_name'] = raw_names['raw_name']\
        .str.replace(r'\s+', ' ', regex=True)\
        .str.strip()\
        .str.title()
    raw_names = pd.merge(
        raw_names.set_index('raw_name'),
        name_dict_raw.set_index('raw_name'),
        left_index=True, right_index=True,
        how='left',
        sort=False
    ).reset_index()
    raw_names = raw_names.dropna(
        subset=['name', 'first_name', key_id], how='any')
    raw_names['name_id'] = raw_names['name_id'].astype(int)
    # key, raw_name, gender, customer_type, name_id, name,
    # last_name, middle_name, first_name, gender_dict, customer_type_dict
    raw_names.loc[
        raw_names['gender'].isna(),
        'gender'
    ] = raw_names['gender_dict']
    raw_names.loc[
        raw_names['customer_type'].isna(),
        'customer_type'
    ] = raw_names['customer_type_dict']

    raw_names = raw_names.drop(
        columns=['gender_dict', 'customer_type_dict', 'raw_name']
    )
    raw_names = raw_names.rename(columns={
        'name': 'raw_name'
    })

    # key, raw_name, gender, customer_type, name_id,
    # last_name, middle_name, first_name, gender, customer_type

    # * split data to choose best name
    dup_keys = raw_names[raw_names[key_id].duplicated()][key_id].unique()
    multi_key_names = raw_names[raw_names[key_id].isin(dup_keys)]
    single_key_names = raw_names[~raw_names[key_id].isin(dup_keys)]

    # * run pipeline best_name
    print(">>> Process multi-key name")
    pre_multi_key_names = CoreBestName(
        multi_key_names, key_id, n_cores=n_cores)

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
    pre_names = pre_names[[
        key_id, "best_name", "raw_name", "name_id",
        "gender", "customer_type", "source_name",
        "similarity_score", "source_best_name"
    ]]

    # * best gender, customer type
    print(">>> Get extra best name info")
    best_name_gender = pre_names[
        (pre_names['best_name'].notna())
        & (pre_names['best_name'] == pre_names['raw_name'])
    ][['best_name', 'gender', 'customer_type']].rename(columns={
        'gender': 'best_gender',
        'customer_type': 'best_customer_type'
    }).drop_duplicates()\
        .dropna(subset=['best_gender', 'best_customer_type'])\
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
    pre_names.loc[
        pre_names['best_customer_type'].isna(),
        'best_customer_type'
    ] = None

    # pre_names = pre_names.rename(
    #     columns={
    #         'raw_name': 'name'
    #     }
    # )
    pre_names = pre_names.drop(columns=['raw_name'])

    # * UNIQUE BEST NAME
    print(">>> Find unique best name for key")
    pre_names = UniqueBestName(
        pre_names,
        key=key_id
    )

    # * Hash name info
    customer_type_dict = pd.read_parquet('/data/fpt/ftel/cads/dep_solution/sa/rst/customer_type_dict.parquet', filesystem=hdfs)
    gender_dict = pd.read_parquet('/data/fpt/ftel/cads/dep_solution/user/nguyennpa2/rst_total/gender_dict.parquet', filesystem=hdfs)
    source_name_dict = pd.read_parquet('/data/fpt/ftel/cads/dep_solution/user/quangvm9/rst_total/source_name_dict.parquet', filesystem=hdfs)

    customer_type_map = dict(zip(
        customer_type_dict['customer_type'],
        customer_type_dict['customer_type_id']
    ))
    for col in ['customer_type', 'best_customer_type', 'unique_customer_type']:
        pre_names[col] = pre_names[col].map(customer_type_map).fillna(-1)
        pre_names[col] = pre_names[col].astype(int)

    gender_map = dict(zip(
        gender_dict['gender'],
        gender_dict['gender_id']
    ))
    for col in ['gender', 'best_gender', 'unique_gender']:
        pre_names[col] = pre_names[col].map(gender_map).fillna(-1)
        pre_names[col] = pre_names[col].astype(int)

    source_name_map = dict(zip(
        source_name_dict['source_name_raw'],
        source_name_dict['source_name']
    ))
    for col in ['source_name', 'source_best_name', 'source_unique_name']:
        pre_names[col] = pre_names[col].map(source_name_map).fillna(pre_names[col])

    # * Change to standard schema
    print(">>> Change to correct schema")
    pre_names = pre_names.drop_duplicates().groupby([
        key_id, 'unique_name', 'unique_gender',
        'unique_customer_type', 'source_unique_name'
    ])\
        .agg(list).reset_index()

    pre_names = pre_names.rename(columns={
        'best_name': 'list_best_name',
        'name_id': 'list_name_id',
        'gender': 'list_gender_id',
        'customer_type': 'list_customer_type_id',
        'source_name': 'list_source_name',
        'similarity_score': 'list_similarity_score',
        'best_gender': 'list_best_gender_id',
        'best_customer_type': 'list_best_customer_type_id',
        'source_best_name': 'list_source_best_name',
    })

    pre_names = pre_names.rename(columns={
        'unique_name': 'best_name',
        'unique_gender': 'best_gender_id',
        'unique_customer_type': 'best_customer_type_id',
        'source_unique_name': 'source_best_name'
    })

    pre_names = pre_names[[
        key_id,
        'best_name', 'best_gender_id', 'best_customer_type_id', 'source_best_name',
        'list_name_id', 'list_gender_id', 'list_customer_type_id', 'list_source_name',
        'list_best_name', 'list_best_gender_id', 'list_best_customer_type_id',
        'list_source_best_name',
        'list_similarity_score'
    ]]

    # Load & concat to new data
    if not init:
        try:
            old_best_name = pd.read_parquet(
                f'{UTILS_PATH}/name_by_{key}_latest.parquet',
                filesystem=hdfs
            )
            pre_names = pd.concat([
                pre_names,
                old_best_name
            ], ignore_index=True)\
                .drop_duplicates(subset=[key_id], keep='first')\
                .sort_values(by=[key_id], ascending=False, ignore_index=True)
        except FileNotFoundError:
            print("No older data found >>> Keep this one as the origin")

    # return
    return pre_names
