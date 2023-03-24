import subprocess
import sys
from datetime import datetime, timedelta
from glob import glob

import pandas as pd
from pyarrow import fs

sys.path.append("/bigdata/fdp/cdp/source/core_profile/preprocess/utils")
from const import CENTRALIZE_PATH, PREPROCESS_PATH, hdfs

# from preprocess_profile import (
#     remove_same_username_email,
#     extracting_pronoun_from_name
# )
# from enhance_profile import enhance_common_profile
from filter_profile import get_difference_data

# function get profile change/new


# def DifferenceProfile(now_df, yesterday_df):
#     difference_df = now_df[~now_df.apply(tuple, 1).isin(
#         yesterday_df.apply(tuple, 1))].copy()
#     return difference_df

# function unify profile


def UnifyFshop(profile_fshop: pd.DataFrame, n_cores: int = 1):
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
    # * Processing info
    print(">>> Processing Info")
    # profile_fshop = profile_fshop.rename(
    #     columns={'customer_type': 'customer_type_fshop'})
    profile_fshop.loc[profile_fshop["gender"] == "-1", "gender"] = None
    profile_fshop.loc[
        profile_fshop["address"].isin(["", "Null", "None", "Test"]), "address"
    ] = None
    profile_fshop.loc[
        profile_fshop["address"].notna()
        & profile_fshop["address"].str.isnumeric(),
        "address",
    ] = None
    profile_fshop.loc[profile_fshop["address"].str.len() < 5, "address"] = None
    profile_fshop["customer_type"] = profile_fshop["customer_type"].replace(
        {"Individual": "Ca nhan", "Company": "Cong ty", "Other": None}
    )

    # * Enhancing common profile
    # profile_fshop = enhance_common_profile(
    #     profile_fshop,
    #     n_cores=n_cores
    # )

    # customer_type extra
    #     print(">>> Extra Customer Type")
    #     profile_fshop.loc[
    #         (profile_fshop['customer_type'] == 'Ca nhan')
    #         & (profile_fshop['customer_type_fshop'].notna()),
    #         'customer_type'
    #     ] = profile_fshop['customer_type_fshop']
    #     profile_fshop = profile_fshop.drop(columns=['customer_type_fshop'])

    #     # drop name is username_email
    #     print(">>> Extra Cleansing Name")
    #     profile_fshop = remove_same_username_email(
    #         profile_fshop,
    #         name_col='name',
    #         email_col='email'
    #     )

    # clean name
    # condition_name =\
    #     (profile_fshop['customer_type'].isin([None, 'Ca nhan', np.nan]))\
    #     & (profile_fshop['name'].notna())
    # profile_fshop = extracting_pronoun_from_name(
    #     profile_fshop,
    #     condition=condition_name,
    #     name_col='name'
    # )

    # is full name
    # print(">>> Checking Full Name")
    # profile_fshop.loc[profile_fshop['last_name'].notna(
    # ) & profile_fshop['first_name'].notna(), 'is_full_name'] = True
    # profile_fshop['is_full_name'] = profile_fshop['is_full_name'].fillna(False)
    # profile_fshop = profile_fshop.drop(
    #     columns=['last_name', 'middle_name', 'first_name'])

    # valid gender by model
    # print(">>> Validating Gender")
    # profile_fshop.loc[
    #     profile_fshop['customer_type'] != 'Ca nhan',
    #     'gender'
    # ] = None
    # profile_fo.loc[profile_fo['gender'].notna() & profile_fo['name'].isna(), 'gender'] = None
    # profile_fshop.loc[
    #     (profile_fshop['gender'].notna())
    #     & (profile_fshop['gender'] != profile_fshop['gender_enrich']),
    #     'gender'
    # ] = None

    # location of shop
    #     shop_fshop = pd.read_parquet('/data/fpt/ftel/cads/dep_solution/user/namdp11/scross_fill/runner/refactor/material/fshop_shop_khanhhb3.parquet',
    #                                  filesystem=hdfs, columns = ['ShopCode', 'LV1_NORM', 'LV2_NORM', 'LV3_NORM']).drop_duplicates()

    print(">>> Processing Address")
    path_shop = [
        f.path
        for f in hdfs.get_file_info(
            fs.FileSelector(
                "/data/fpt/ftel/cads/dep_solution/user/namdp11/scross_fill/runner/refactor/material/fshop_shop_khanhhb3_ver2.parquet/"
            )
        )
    ][-1]
    shop_fshop = pd.read_parquet(
        path_shop,
        filesystem=hdfs,
        columns=["ShopCode", "Level1Norm", "Level2Norm", "Level3Norm"],
    ).drop_duplicates()
    shop_fshop.columns = ["shop_code", "city", "district", "ward"]
    shop_fshop["shop_code"] = shop_fshop["shop_code"].astype(str)

    transaction_paths = sorted(
        glob("/bigdata/fdp/frt/data/posdata/ict/pos_ordr/*")
    )
    transaction_fshop = pd.DataFrame()
    for path in transaction_paths:
        df = pd.read_parquet(path)
        df = df[["CardCode", "ShopCode", "Source"]].drop_duplicates()
        df.columns = ["cardcode", "shop_code", "source"]
        df["shop_code"] = df["shop_code"].astype(str)

        df = pd.merge(
            df.set_index("shop_code"),
            shop_fshop.set_index("shop_code"),
            left_index=True,
            right_index=True,
            how="left",
            sort=False,
        ).reset_index()
        df = df.sort_values(by=["cardcode", "source"], ascending=True)
        df = df.drop_duplicates(subset=["cardcode"], keep="last")
        df = df[
            ["cardcode", "city", "district", "ward", "source"]
        ].reset_index(drop=True)
        transaction_fshop = pd.concat(
            [transaction_fshop, df], ignore_index=True
        )

    transaction_fshop = transaction_fshop.sort_values(
        by=["cardcode", "source"], ascending=True
    )
    transaction_fshop = transaction_fshop.drop_duplicates(
        subset=["cardcode"], keep="last"
    )
    transaction_fshop.to_parquet(
        "/data/fpt/ftel/cads/dep_solution/user/namdp11/scross_fill/runner/refactor/material/fshop_location_latest_khanhhb3.parquet",
        index=False,
        filesystem=hdfs,
    )

    # location of profile
    profile_location_fshop = pd.read_parquet(
        "/data/fpt/ftel/cads/dep_solution/sa/cdp/data/fshop_address_latest.parquet",
        columns=["CardCode", "Address", "Ward", "District", "City", "Street"],
        filesystem=hdfs,
    )
    profile_location_fshop.columns = [
        "cardcode",
        "address",
        "ward",
        "district",
        "city",
        "street",
    ]
    profile_location_fshop = profile_location_fshop.rename(
        columns={"cardcode": "uid"}
    )
    profile_location_fshop.loc[
        profile_location_fshop["address"].isin(["", "Null", "None", "Test"]),
        "address",
    ] = None
    profile_location_fshop.loc[
        profile_location_fshop["address"].str.len() < 5, "address"
    ] = None
    profile_location_fshop["address"] = (
        profile_location_fshop["address"].str.strip().replace(dict_trash)
    )
    profile_location_fshop = profile_location_fshop.drop_duplicates(
        subset=["uid"], keep="first"
    )

    latest_location_fshop = pd.read_parquet(
        "/data/fpt/ftel/cads/dep_solution/user/namdp11/scross_fill/runner/refactor/material/fshop_location_latest_khanhhb3.parquet",
        filesystem=hdfs,
    )
    latest_location_fshop = latest_location_fshop.drop(columns=["source"])
    latest_location_fshop = latest_location_fshop.rename(
        columns={"cardcode": "uid"}
    )
    latest_location_fshop["ward"] = None

    # source address
    latest_location_fshop.loc[
        latest_location_fshop["city"].notna(), "source_city"
    ] = "FSHOP from shop"
    latest_location_fshop.loc[
        latest_location_fshop["district"].notna(), "source_district"
    ] = "FSHOP from shop"
    latest_location_fshop.loc[
        latest_location_fshop["ward"].notna(), "source_ward"
    ] = "FSHOP from shop"

    profile_location_fshop.loc[
        profile_location_fshop["city"].notna(), "source_city"
    ] = "FSHOP from profile"
    profile_location_fshop.loc[
        profile_location_fshop["district"].notna(), "source_district"
    ] = "FSHOP from profile"
    profile_location_fshop.loc[
        profile_location_fshop["ward"].notna(), "source_ward"
    ] = "FSHOP from profile"

    # from shop: miss ward & district & city
    profile_location_fshop_bug = profile_location_fshop[
        (profile_location_fshop["city"].isna())
        & (profile_location_fshop["district"].isna())
        & (profile_location_fshop["ward"].isna())
    ]
    profile_location_fshop_bug = profile_location_fshop_bug[["uid", "address"]]
    profile_location_fshop_bug = pd.merge(
        profile_location_fshop_bug.set_index("uid"),
        latest_location_fshop.set_index("uid"),
        left_index=True,
        right_index=True,
        how="left",
        sort=False,
    ).reset_index()
    profile_location_fshop = profile_location_fshop[
        ~profile_location_fshop["uid"].isin(profile_location_fshop_bug["uid"])
    ]
    profile_location_fshop = pd.concat(
        [profile_location_fshop, profile_location_fshop_bug], ignore_index=True
    )

    # from shop: miss district & city
    profile_location_fshop_bug = profile_location_fshop[
        profile_location_fshop["city"].isna()
        & profile_location_fshop["district"].isna()
    ]
    profile_location_fshop_bug = profile_location_fshop_bug.drop(
        columns=["city", "source_city", "district", "source_district"]
    )
    temp_latest_location_fshop = latest_location_fshop[
        ["uid", "city", "source_city", "district", "source_district"]
    ]
    profile_location_fshop_bug = pd.merge(
        profile_location_fshop_bug.set_index("uid"),
        temp_latest_location_fshop.set_index("uid"),
        left_index=True,
        right_index=True,
        how="left",
        sort=False,
    ).reset_index()
    profile_location_fshop = profile_location_fshop[
        ~profile_location_fshop["uid"].isin(profile_location_fshop_bug["uid"])
    ]
    profile_location_fshop = pd.concat(
        [profile_location_fshop, profile_location_fshop_bug], ignore_index=True
    )

    # from shop: miss city
    profile_location_fshop_bug = profile_location_fshop[
        profile_location_fshop["city"].isna()
        & profile_location_fshop["district"].notna()
    ]
    profile_location_fshop_bug = profile_location_fshop_bug.drop(
        columns=["city", "source_city"]
    )
    temp_latest_location_fshop = latest_location_fshop[
        ["uid", "district", "city", "source_city"]
    ]
    profile_location_fshop_bug = pd.merge(
        profile_location_fshop_bug.set_index("uid"),
        temp_latest_location_fshop.set_index("uid"),
        left_index=True,
        right_index=True,
        how="left",
        sort=False,
    ).reset_index()
    profile_location_fshop = profile_location_fshop[
        ~profile_location_fshop["uid"].isin(profile_location_fshop_bug["uid"])
    ]
    profile_location_fshop = pd.concat(
        [profile_location_fshop, profile_location_fshop_bug], ignore_index=True
    )

    # from shop: miss district
    profile_location_fshop_bug = profile_location_fshop[
        profile_location_fshop["city"].notna()
        & profile_location_fshop["district"].isna()
    ]
    profile_location_fshop_bug = profile_location_fshop_bug.drop(
        columns=["district", "source_district"]
    )
    temp_latest_location_fshop = latest_location_fshop[
        ["uid", "city", "district", "source_district"]
    ]
    profile_location_fshop_bug = pd.merge(
        profile_location_fshop_bug.set_index("uid"),
        temp_latest_location_fshop.set_index("uid"),
        left_index=True,
        right_index=True,
        how="left",
        sort=False,
    ).reset_index()
    profile_location_fshop = profile_location_fshop[
        ~profile_location_fshop["uid"].isin(profile_location_fshop_bug["uid"])
    ]
    profile_location_fshop = pd.concat(
        [profile_location_fshop, profile_location_fshop_bug], ignore_index=True
    )

    # normlize address
    profile_fshop["address"] = (
        profile_fshop["address"].str.strip().replace(dict_trash)
    )
    profile_fshop = profile_fshop.drop(columns=["city"])
    profile_fshop = profile_fshop.merge(
        profile_location_fshop, how="left", on=["uid", "address"]
    )

    profile_fshop.loc[profile_fshop["street"].isna(), "street"] = None
    profile_fshop.loc[profile_fshop["ward"].isna(), "ward"] = None
    profile_fshop.loc[profile_fshop["district"].isna(), "district"] = None
    profile_fshop.loc[profile_fshop["city"].isna(), "city"] = None

    # full address
    columns = ["street", "ward", "district", "city"]
    profile_fshop["address"] = (
        profile_fshop[columns]
        .fillna("")
        .agg(", ".join, axis=1)
        .str.replace("(?<![a-zA-Z0-9]),", "", regex=True)
        .str.replace("-(?![a-zA-Z0-9])", "", regex=True)
    )
    profile_fshop["address"] = (
        profile_fshop["address"].str.strip(", ").str.strip(",").str.strip()
    )
    profile_fshop["address"] = (
        profile_fshop["address"].str.strip().replace(dict_trash)
    )
    profile_fshop.loc[
        profile_fshop["address"].notna(), "source_address"
    ] = profile_fshop["source_city"]

    # unit_address
    profile_fshop = profile_fshop.rename(columns={"street": "unit_address"})
    profile_fshop.loc[
        profile_fshop["unit_address"].notna(), "source_unit_address"
    ] = "FSHOP from profile"

    # add info
    print(">>> Adding Temp Info")
    # profile_fshop['birthday'] = None
    columns = [
        "uid",
        "phone",
        "email",
        "card",
        "name",
        "gender",
        "birthday",
        "age",
        "customer_type",
        "address",
        "city",
        "district",
        "ward",
        "street",
        "source_address",
        "source_city",
        "source_district",
        "source_ward",
        "source_street",
        "source",
    ]
    profile_fshop = profile_fshop[columns]

    # # Fill 'Ca nhan'
    # profile_fshop.loc[
    #     (profile_fshop['name'].notna())
    #     & (profile_fshop['customer_type'].isna()),
    #     'customer_type'
    # ] = 'Ca nhan'

    # return
    return profile_fshop


# function update profile (unify)


def UpdateUnifyFshop(now_str: str, n_cores: int = 1):
    # VARIABLES
    # raw_path = ROOT_PATH + '/raw'
    # unify_path = ROOT_PATH + '/pre'
    f_group = "frt_fshop"
    yesterday_str = (
        datetime.strptime(now_str, "%Y-%m-%d") - timedelta(days=1)
    ).strftime("%Y-%m-%d")

    # load profile (yesterday, now)
    print(">>> Loading today and yesterday profile")
    info_columns = [
        "uid",
        "phone",
        "email",
        "card",
        "name",
        "gender",
        "birthday",
        "age",
        "customer_type",
        "address",
        "city",
        "district",
        "ward",
        "street",
        "source",
    ]
    now_profile = pd.read_parquet(
        f"{CENTRALIZE_PATH}/{f_group}.parquet/d={now_str}",
        filesystem=hdfs,
        columns=info_columns,
    )
    yesterday_profile = pd.read_parquet(
        f"{CENTRALIZE_PATH}/{f_group}.parquet/d={yesterday_str}",
        filesystem=hdfs,
        columns=info_columns,
    )

    # get profile change/new
    print(">>> Filtering new profile")
    difference_profile = get_difference_data(now_profile, yesterday_profile)
    print(f"Number of new profile {difference_profile.shape}")

    # update profile
    profile_unify = pd.read_parquet(
        f"{PREPROCESS_PATH}/{f_group}.parquet/d={yesterday_str}",
        filesystem=hdfs,
    )
    if not difference_profile.empty:
        # get profile unify (old + new)
        new_profile_unify = UnifyFshop(difference_profile, n_cores=n_cores)

        # synthetic profile
        profile_unify = pd.concat(
            [new_profile_unify, profile_unify], ignore_index=True
        )

    # arrange columns
    print(">>> Re-Arranging Columns")
    columns = [
        "uid",
        "phone",
        "email",
        "card",
        "name",
        "gender",
        "birthday",
        "age",
        "customer_type",
        "address",
        "city",
        "district",
        "ward",
        "street",
        "source_address",
        "source_city",
        "source_district",
        "source_ward",
        "source_street",
        "source",
    ]

    profile_unify = profile_unify[columns]
    # profile_unify['is_phone_valid'] =\
    #     profile_unify['is_phone_valid'].fillna(False)
    # profile_unify['is_email_valid'] =\
    #     profile_unify['is_email_valid'].fillna(False)
    # profile_unify = profile_unify.drop_duplicates(
    #     subset=['uid', 'phone_raw', 'email_raw'],
    #     keep='first'
    # )

    # Type casting for saving
    print(">>> Process casting columns...")
    # profile_unify['uid'] = profile_unify['uid'].astype(str)
    profile_unify["birthday"] = profile_unify["birthday"].astype(
        "datetime64[s]"
    )

    # save
    print(f"Checking {f_group} data for {now_str}...")
    f_group_path = f"{PREPROCESS_PATH}/{f_group}.parquet"
    proc = subprocess.Popen(
        ["hdfs", "dfs", "-test", "-e", f_group_path + f"/d={now_str}"]
    )
    proc.communicate()
    if proc.returncode == 0:
        print("Data already existed, Removing...")
        subprocess.run(
            ["hdfs", "dfs", "-rm", "-r", f_group_path + f"/d={now_str}"]
        )

    profile_unify["d"] = now_str
    profile_unify.to_parquet(
        f_group_path,
        filesystem=hdfs,
        index=False,
        partition_cols="d",
        coerce_timestamps="us",
        allow_truncated_timestamps=True,
    )


if __name__ == "__main__":
    DAY = sys.argv[1]
    UpdateUnifyFshop(DAY, n_cores=10)
