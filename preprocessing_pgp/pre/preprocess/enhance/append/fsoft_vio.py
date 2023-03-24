import subprocess
import sys
from datetime import datetime, timedelta

import pandas as pd

sys.path.append("/bigdata/fdp/cdp/source/core_profile/preprocess/utils")
# from enhance_profile import enhance_common_profile
from const import CENTRALIZE_PATH, PREPROCESS_PATH, hdfs

# from preprocess_profile import (
#     remove_same_username_email,
#     extracting_pronoun_from_name
# )
from filter_profile import get_difference_data

# function get profile change/new
# def DifferenceProfile(now_df, yesterday_df):
#     difference_df = now_df[~now_df.apply(tuple,1).isin(yesterday_df.apply(tuple,1))].copy()
#     return difference_df

# function unify profile


def UnifyFsoft(profile_fsoft: pd.DataFrame, n_cores: int = 1):
    # VARIABLE
    # dict_trash = {
    #     '': None, 'Nan': None, 'nan': None,
    #     'None': None, 'none': None, 'Null': None,
    #     'null': None, "''": None
    # }

    # * Processing info
    print(">>> Processing Info")
    # profile_fsoft.loc[profile_fsoft['gender'] == '-1', 'gender'] = None
    profile_fsoft.loc[
        profile_fsoft["address"].isin(["", "Null", "None", "Test"]), "address"
    ] = None
    profile_fsoft.loc[
        profile_fsoft["address"].notna()
        & profile_fsoft["address"].str.isnumeric(),
        "address",
    ] = None
    profile_fsoft.loc[profile_fsoft["address"].str.len() < 5, "address"] = None
    profile_fsoft["customer_type"] = profile_fsoft["customer_type"].replace(
        {"Individual": "Ca nhan", "Company": "Cong ty", "Other": None}
    )

    # * Enhancing common profile
    # profile_fsoft = enhance_common_profile(
    #     profile_fsoft,
    #     n_cores=n_cores
    # )

    # customer_type extra
    #     print(">>> Extra Customer Type")
    #     profile_fsoft.loc[
    #         (profile_fsoft['customer_type'] == 'Ca nhan')
    #         & (profile_fsoft['customer_type_fsoft'].notna()),
    #         'customer_type'
    #     ] = profile_fsoft['customer_type_fsoft']
    #     profile_fsoft = profile_fsoft.drop(columns=['customer_type_fsoft'])

    #     # drop name is username_email
    #     print(">>> Extra Cleansing Name")
    #     profile_fsoft = remove_same_username_email(
    #         profile_fsoft,
    #         name_col='name',
    #         email_col='email'
    #     )

    #     # clean name, extract pronoun
    #     condition_name = (profile_fsoft['customer_type'] == 'Ca nhan')\
    #         & (profile_fsoft['name'].notna())
    #     profile_fsoft = extracting_pronoun_from_name(
    #         profile_fsoft,
    #         condition=condition_name,
    #         name_col='name',
    #     )

    #     # is full name
    #     print(">>> Checking Full Name")
    #     profile_fsoft.loc[profile_fsoft['last_name'].notna(
    #     ) & profile_fsoft['first_name'].notna(), 'is_full_name'] = True
    #     profile_fsoft['is_full_name'] = profile_fsoft['is_full_name'].fillna(False)
    #     profile_fsoft = profile_fsoft.drop(
    #         columns=['last_name', 'middle_name', 'first_name'])

    #     # valid gender by model
    #     print(">>> Validating Gender")
    #     profile_fsoft.loc[
    #         profile_fsoft['customer_type'] != 'Ca nhan',
    #         'gender'
    #     ] = None
    #     # profile_fo.loc[profile_fo['gender'].notna() & profile_fo['name'].isna(), 'gender'] = None
    #     profile_fsoft.loc[
    #         (profile_fsoft['gender'].notna())
    #         & (profile_fsoft['gender'] != profile_fsoft['gender_enrich']),
    #         'gender'
    #     ] = None

    #     # normalize address
    #     print(">>> Processing Address")
    #     profile_fsoft['address'] = profile_fsoft['address'].str.strip().replace(
    #         dict_trash)
    #     profile_fsoft['street'] = None
    #     profile_fsoft['ward'] = None

    # ## full address
    # columns = ['street', 'ward', 'district', 'city']
    # profile_fsoft['address'] = profile_fsoft[columns].fillna('').agg(', '.join, axis=1).str.replace('(?<![a-zA-Z0-9]),', '', regex=True).str.replace('-(?![a-zA-Z0-9])', '', regex=True)
    # profile_fsoft['address'] = profile_fsoft['address'].str.strip(', ').str.strip(',').str.strip()
    # profile_fsoft['address'] = profile_fsoft['address'].str.strip().replace(dict_trash)
    # profile_fsoft.loc[profile_fsoft['address'].notna(), 'source_address'] = profile_fsoft['source_city']

    # unit_address
    # profile_fsoft = profile_fsoft.rename(columns={'street': 'unit_address'})
    # profile_fsoft.loc[profile_fsoft['unit_address'].notna(
    # ), 'source_unit_address'] = 'FSOFT from profile'
    # profile_fsoft.loc[profile_fsoft['ward'].notna(
    # ), 'source_ward'] = 'FSOFT from profile'
    # profile_fsoft.loc[profile_fsoft['district'].notna(
    # ), 'source_district'] = 'FSOFT from profile'
    # profile_fsoft.loc[profile_fsoft['city'].notna(
    # ), 'source_city'] = 'FSOFT from profile'
    # profile_fsoft.loc[profile_fsoft['address'].notna(
    # ), 'source_address'] = 'FSOFT from profile'

    # add info
    print(">>> Adding Temp Info")
    profile_fsoft[
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
    profile_fsoft = profile_fsoft[columns]

    # Fill 'Ca nhan'
    # profile_fsoft.loc[
    #     (profile_fsoft['name'].notna())
    #     & (profile_fsoft['customer_type'].isna()),
    #     'customer_type'
    # ] = 'Ca nhan'
    # return
    return profile_fsoft


# function update profile (unify)


def UpdateUnifyFsoft(now_str: str, n_cores: int = 1):
    # VARIABLES
    f_group = "fsoft_vio"
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
        new_profile_unify = UnifyFsoft(difference_profile, n_cores=n_cores)

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
    #     profile_unify['is_phone_valid'] =\
    #         profile_unify['is_phone_valid'].fillna(False)
    #     profile_unify['is_email_valid'] =\
    #         profile_unify['is_email_valid'].fillna(False)
    #     profile_unify = profile_unify.drop_duplicates(
    #         subset=['uid', 'phone_raw', 'email_raw'], keep='first')

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
    UpdateUnifyFsoft(DAY, n_cores=10)
