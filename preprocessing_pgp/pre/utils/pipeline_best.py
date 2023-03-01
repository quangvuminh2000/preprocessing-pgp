"""
Module contains pipeline for best profile
"""

from const import (
    hdfs,
    ROOT_PATH
)
import sys
import multiprocessing as mp
from difflib import SequenceMatcher

import pandas as pd
from tqdm import tqdm
from unidecode import unidecode
from preprocessing_pgp.name.split_name import NameProcess

sys.path.append('/bigdata/fdp/cdp/cdp_pages/scripts_hdfs/pre/utils/new')


def LoadNameGender(
    date_str: str,
    key: str = "phone"
):
    # create
    pre_path = ROOT_PATH + "/pre"
    cttvs = ["fo", "fplay", "ftel", "sendo", "fshop", "longchau", "fsoft"]
    if key == "phone":  # Credit data not having email
        cttvs = cttvs + ["credit"]
    name_gender = pd.DataFrame()

    # loop
    for name_cttv in tqdm(cttvs):
        # read data
        data_cttv = pd.read_parquet(
            f"{pre_path}/{name_cttv}_new.parquet/d={date_str}",
            filesystem=hdfs,
            columns=[key, "name", "customer_type", "gender"],
        )

        # filters
        data_cttv = data_cttv[
            data_cttv[key].notna()
            & data_cttv["name"].notna()
        ]
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

    print(">> Skip/Filter name")

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
    print(">> Create group_id")

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
    print(">> 1 first_name - n last_name")

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
        (map_names_n_df["raw_name"] !=
         map_names_n_df["raw_name"].apply(unidecode))
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
    print(">> 1 first_name - 1 last_name")

    map_single_name_df = single_last_name_df[["group_id"]].drop_duplicates()
    for element_name in ["last_name", "middle_name", "first_name"]:
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
    names_df = pd.concat(
        [single_last_name_df, multi_last_names_df], ignore_index=True)

    # Calculate simility_score
    name_list_1 = list(names_df["raw_name"].unique())
    name_list_2 = list(names_df["best_name"].unique())
    map_element_name = pd.DataFrame()
    map_element_name["raw_name"] = list(set(name_list_1) | set(name_list_2))

    with mp.Pool(n_cores) as pool:
        map_element_name[["last_name", "middle_name", "first_name"]] =\
            pool.map(name_process.SplitName, map_element_name["raw_name"])

    for flag in ["raw", "best"]:
        temp = map_element_name
        temp.columns = [
            f"{flag}_name",
            f"{flag}_last_name",
            f"{flag}_middle_name",
            f"{flag}_first_name",
        ]
        names_df = pd.merge(
            names_df.set_index(f'{flag}_name'),
            temp.set_index(f'{flag}_name'),
            left_index=True, right_index=True,
            how='left',
            sort=True
        ).reset_index()

    # similar score by element
    for element_name in ["last_name", "middle_name", "first_name"]:
        # split data to compare
        condition_compare = (
            names_df[f"raw_{element_name}"].notna()
            & names_df[f"best_{element_name}"].notna()
        )
        compare_names_df = names_df[condition_compare]
        not_compare_names_df = names_df[~condition_compare]

        # compare raw with best
        compare_names_df[f"similar_{element_name}"] = compare_names_df[
            f"raw_{element_name}"
        ].apply(unidecode) == compare_names_df[f"best_{element_name}"].apply(unidecode)
        compare_names_df[f"similar_{element_name}"] = compare_names_df[
            f"similar_{element_name}"
        ].astype(int)

        not_compare_names_df[f"similar_{element_name}"] = 1

        # concat
        names_df = pd.concat(
            [compare_names_df, not_compare_names_df], ignore_index=True
        )

    weights = [0.25, 0.25, 0.5]
    names_df["simility_score"] = (
        weights[0] * names_df["similar_last_name"]
        + weights[1] * names_df["similar_middle_name"]
        + weights[2] * names_df["similar_first_name"]
    )

    print(">> Simility_score")

    # Postprocess
    pre_names_df = names_df[
        ["group_id", "raw_name", "best_name", "simility_score"]
    ]
    pre_names_df[key] = pre_names_df["group_id"].str.split("-").str[0]
    pre_names_df = pre_names_df.drop(columns=["group_id"])

    pre_names_df = pd.concat([
        pre_names_df,
        skip_names_df
    ], ignore_index=True)
    pre_names_df.loc[pre_names_df["best_name"].isna(), "best_name"] = pre_names_df[
        "raw_name"
    ]
    pre_names_df.loc[pre_names_df["simility_score"].isna(),
                     "simility_score"] = 1

    print(">> Postprocess")

    # Merge
    pre_names_n = pd.merge(
        raw_names_n.set_index([key, 'raw_name']),
        pre_names_df.set_index([key, 'raw_name']),
        left_index=True, right_index=True,
        how='left',
        sort=True
    ).reset_index()
    pre_names_n.loc[pre_names_n["best_name"].isna(), "best_name"] = pre_names_n[
        "raw_name"
    ]
    pre_names_n.loc[pre_names_n["simility_score"].isna(), "simility_score"] = 1

    # Find source best_name
    pre_names_n["score_by_best"] = pre_names_n[["raw_name", "best_name"]].apply(
        lambda row: SequenceMatcher(None, row.raw_name, row.best_name).ratio(), axis=1
    )
    map_source_best_name = (
        pre_names_n.sort_values(
            by=[key, "best_name", "score_by_best"], ascending=False)
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
    pre_names_n = pre_names_n.merge(
        map_source_best_name, how="left", on=[key, "best_name"]
    )
    pre_names_n = pre_names_n.drop(columns=["score_by_best"])

    # Return
    return pre_names_n


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
    pre_multi_key_names = CoreBestName(multi_key_names, key, n_cores=n_cores)

    # fake best_name
    pre_single_key_names = single_key_names
    pre_single_key_names["best_name"] = pre_single_key_names["raw_name"]
    pre_single_key_names["simility_score"] = 1
    pre_single_key_names["source_best_name"] = pre_single_key_names["source_name"]

    # concat
    pre_names = pd.concat(
        [pre_single_key_names, pre_multi_key_names], ignore_index=True)

    # best gender
    best_name_gender = pre_names[
        (pre_names['best_name'].notna())
        & (pre_names['best_name'] == pre_names['raw_name'])
    ][['best_name', 'gender']].rename(columns={
        'gender': 'best_gender'
    }).drop_duplicates()
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

    # rename
    pre_names = pre_names.rename(
        columns={
            'raw_name': 'name'
        }
    )

    # return
    return pre_names
