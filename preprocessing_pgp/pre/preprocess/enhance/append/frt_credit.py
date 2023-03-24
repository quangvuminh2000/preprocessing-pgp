import subprocess
import sys
from datetime import datetime, timedelta

import pandas as pd

sys.path.append("/bigdata/fdp/cdp/source/core_profile/preprocess/utils")
from const import CENTRALIZE_PATH, PREPROCESS_PATH, hdfs

# from preprocess_profile import (
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


def UnifyCredit(profile_credit: pd.DataFrame, n_cores: int = 1):
    # * Processing info
    print(">>> Processing Info")
    # profile_credit = profile_credit.rename(
    #     columns={
    #         'customer_type': 'customer_type_credit',
    #     }
    # )
    profile_credit.loc[profile_credit["gender"] == "-1", "gender"] = None
    profile_credit.loc[
        profile_credit["address"].isin(["", "Null", "None", "Test"]), "address"
    ] = None
    profile_credit.loc[
        profile_credit["address"].notna()
        & profile_credit["address"].str.isnumeric(),
        "address",
    ] = None
    profile_credit.loc[
        profile_credit["address"].str.len() < 5, "address"
    ] = None
    profile_credit["customer_type"] = profile_credit["customer_type"].replace(
        {"Individual": "Ca nhan", "Company": "Cong ty", "Other": None}
    )

    # * Enhancing common profile
    #     profile_credit = enhance_common_profile(
    #         profile_credit,
    #         enhance_email=False,
    #         n_cores=n_cores
    #     )

    #     # customer_type extra
    #     print(">>> Extra Customer Type")
    #     profile_credit.loc[
    #         (profile_credit['customer_type'] == 'Ca nhan')
    #         & (profile_credit['customer_type_credit'].notna()),
    #         'customer_type'
    #     ] = profile_credit['customer_type_credit']
    #     profile_credit = profile_credit.drop(columns=['customer_type_credit'])

    #     # clean name
    #     condition_name =\
    #         (profile_credit['customer_type'].isin([None, 'Ca nhan', np.nan]))\
    #         & (profile_credit['name'].notna())
    #     profile_credit = extracting_pronoun_from_name(
    #         profile_credit,
    #         condition=condition_name,
    #         name_col='name',
    #     )

    #     # is full name
    #     print(">>> Checking Full Name")
    #     profile_credit.loc[profile_credit['last_name'].notna(
    #     ) & profile_credit['first_name'].notna(), 'is_full_name'] = True
    #     profile_credit['is_full_name'] = profile_credit['is_full_name'].fillna(
    #         False)
    #     profile_credit = profile_credit.drop(
    #         columns=['last_name', 'middle_name', 'first_name'])

    #     # valid gender by model
    #     print(">>> Validating Gender")
    #     profile_credit.loc[
    #         profile_credit['customer_type'] != 'Ca nhan',
    #         'gender'
    #     ] = None

    #     profile_credit.loc[
    #         (profile_credit['gender'].notna())
    #         & (profile_credit['gender'] != profile_credit['gender_enrich']),
    #         'gender'
    #     ] = None

    # normlize address
    # print(">>> Processing Address")
    # profile_credit['address'] = profile_credit['address'].str.strip().replace(
    #     dict_trash)
    # profile_credit['street'] = None
    # profile_credit['ward'] = None
    # profile_credit['district'] = None
    # profile_credit['city'] = None

    # unit_address
    # profile_credit = profile_credit.rename(columns={'street': 'unit_address'})
    # profile_credit.loc[profile_credit['unit_address'].notna(
    # ), 'source_unit_address'] = 'CREDIT from profile'
    # profile_credit.loc[profile_credit['ward'].notna(
    # ), 'source_ward'] = 'CREDIT from profile'
    # profile_credit.loc[profile_credit['district'].notna(
    # ), 'source_district'] = 'CREDIT from profile'
    # profile_credit.loc[profile_credit['city'].notna(
    # ), 'source_city'] = 'CREDIT from profile'
    # profile_credit.loc[profile_credit['address'].notna(
    # ), 'source_address'] = 'CREDIT from profile'

    # add info
    print(">>> Adding Temp Info")
    profile_credit[
        [
            "source_address",
            "source_city",
            "source_district",
            "source_ward",
            "source_street",
        ]
    ] = None
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

    profile_credit = profile_credit[columns]

    # Fill 'Ca nhan'
    # profile_credit.loc[
    #     (profile_credit['name'].notna())
    #     & (profile_credit['customer_type'].isna()),
    #     'customer_type'
    # ] = 'Ca nhan'

    # return
    return profile_credit


# function update profile (unify)


def UpdateUnifyCredit(now_str: str, n_cores: int = 1):
    # VARIABLES
    f_group = "frt_credit"
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
        new_profile_unify = UnifyCredit(difference_profile, n_cores=n_cores)

        # synthetic profile
        profile_unify = pd.concat(
            [new_profile_unify, profile_unify], ignore_index=True
        )

    # arrange columns
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
    # profile_unify['is_phone_valid'] = profile_unify['is_phone_valid'].fillna(
    #     False)
    profile_unify = profile_unify.drop_duplicates(
        subset=["uid", "phone"], keep="first"
    )

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
    UpdateUnifyCredit(DAY, n_cores=5)
