import subprocess
import sys
from datetime import datetime, timedelta

import pandas as pd
from pyarrow import fs

sys.path.append("/bigdata/fdp/cdp/source/core_profile/preprocess/utils")
# from enhance_profile import enhance_common_profile
from const import CENTRALIZE_PATH, PREPROCESS_PATH, UTILS_PATH, hdfs

# from preprocess_profile import (
#     remove_same_username_email,
#     extracting_pronoun_from_name
# )
from filter_profile import get_difference_data

# function get profile change/new


# def DifferenceProfile(now_df, yesterday_df):
#     difference_df = now_df[~now_df.apply(tuple, 1).isin(
#         yesterday_df.apply(tuple, 1))].copy()
#     return difference_df

# function unify profile


def UnifyFplay(profile_fplay: pd.DataFrame, n_cores: int = 1):
    # * Processing info
    print(">>> Processing Info")
    # profile_fplay = profile_fplay.rename(columns={'uid': 'user_id'})
    profile_fplay = profile_fplay.sort_values(by=["uid"], ascending=False)
    profile_fplay = profile_fplay.drop_duplicates(subset=["uid"], keep="first")

    # * Enhancing common profile
    #     profile_fplay = enhance_common_profile(
    #         profile_fplay,
    #         n_cores=n_cores
    #     )

    #     profile_fplay = profile_fplay.rename(columns={'gender_enrich': 'gender'})

    # drop name is username_email
    # print(">>> Extra Cleansing Name")
    # profile_fplay = remove_same_username_email(
    #     profile_fplay,
    #     name_col='name',
    #     email_col='email'
    # )

    # clean name, extract_pronoun
    #     condition_name = (profile_fplay['customer_type'] == 'Ca nhan')\
    #         & (profile_fplay['name'].notna())

    #     profile_fplay = extracting_pronoun_from_name(
    #         profile_fplay,
    #         condition=condition_name,
    #         name_col='name',
    #     )

    # is full name
    # print(">>> Checking Full Name")
    # profile_fplay.loc[profile_fplay['last_name'].notna(
    # ) & profile_fplay['first_name'].notna(), 'is_full_name'] = True
    # profile_fplay['is_full_name'] = profile_fplay['is_full_name'].fillna(False)
    # profile_fplay = profile_fplay.drop(
    #     columns=['last_name', 'middle_name', 'first_name'])

    # add info
    print(">>> Adding Temp Info")
    # profile_fplay['birthday'] = None
    # profile_fplay['address'] = None
    # profile_fplay['unit_address'] = None
    # profile_fplay['ward'] = None
    # profile_fplay['district'] = None
    # profile_fplay['city'] = None
    profile_fplay[
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
    profile_fplay = profile_fplay[columns]
    # profile_fplay = profile_fplay.rename(columns={'user_id': 'uid'})

    # Fill 'Ca nhan'
    # profile_fplay.loc[
    #     (profile_fplay['name'].notna())
    #     & (profile_fplay['customer_type'].isna()),
    #     'customer_type'
    # ] = 'Ca nhan'

    # return
    return profile_fplay


# function update profile (unify)


def UpdateUnifyFplay(now_str: str, n_cores: int = 1):
    # VARIABLES
    f_group = "ftel_fplay"
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
        new_profile_unify = UnifyFplay(difference_profile, n_cores=n_cores)

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
    # profile_unify['is_email_valid'] = profile_unify['is_email_valid'].fillna(
    #     False)
    # profile_unify = profile_unify.drop_duplicates(
    #     subset=['uid', 'phone_raw', 'email_raw'], keep='first')

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


# function update ip (most)


def UnifyLocationIpFplay():
    # MOST LOCATION IP
    dict_ip_path = "/data/fpt/ftel/cads/dep_solution/user/namdp11/scross_fill/runner/ip/dictionary"
    log_ip_path = "/data/fpt/ftel/cads/dep_solution/user/namdp11/scross_fill/runner/ip/fplay"

    ip_location1 = pd.read_parquet(
        f"{dict_ip_path}/ip_location_batch_1.parquet", filesystem=hdfs
    )
    ip_location2 = pd.read_parquet(
        f"{dict_ip_path}/ip_location_batch_2.parquet", filesystem=hdfs
    )
    ip_location = ip_location1.append(ip_location2, ignore_index=True)
    ip_location = ip_location[["ip", "name_province", "name_district"]].copy()

    # update ip
    def IpFplay(date):
        date_str = date.strftime("%Y-%m-%d")
        try:
            # load log ip
            log_df = pd.read_parquet(
                f"/data/fpt/ftel/fplay/dwh/ds_network.parquet/d={date_str}",
                filesystem=hdfs,
                columns=["user_id", "ip", "isp", "network_type"],
            ).drop_duplicates()
            log_df["date"] = date_str
            log_df.to_parquet(
                f"{log_ip_path}/ip_{date_str}.parquet",
                index=False,
                filesystem=hdfs,
            )

            # add location
            location_df = log_df.merge(ip_location, how="left", on="ip")
            location_df.to_parquet(
                f"{log_ip_path}/location/ip_{date_str}.parquet",
                index=False,
                filesystem=hdfs,
            )

        except:
            print("IP-FPLAY Fail: {}".format(date_str))

    start_date = sorted(
        [f.path for f in hdfs.get_file_info(fs.FileSelector(log_ip_path))]
    )[-2][-18:-8]
    end_date = (datetime.today() - timedelta(days=1)).strftime("%Y-%m-%d")
    dates = pd.date_range(start_date, end_date, freq="D")

    for date in dates:
        IpFplay(date)

    # stats location ip
    logs_ip_path = sorted(
        [
            f.path
            for f in hdfs.get_file_info(
                fs.FileSelector(f"{log_ip_path}/location/")
            )
        ]
    )[-180:]
    ip_fplay = pd.read_parquet(logs_ip_path, filesystem=hdfs)
    stats_ip_fplay = (
        ip_fplay.groupby(by=["user_id", "name_province", "name_district"])[
            "date"
        ]
        .agg(num_date="count")
        .reset_index()
    )
    stats_ip_fplay = stats_ip_fplay.sort_values(
        by=["user_id", "num_date"], ascending=False
    )
    most_ip_fplay = stats_ip_fplay.drop_duplicates(
        subset=["user_id"], keep="first"
    )
    most_ip_fplay.to_parquet(
        f"{UTILS_PATH}/fplay_location_most.parquet",
        index=False,
        filesystem=hdfs,
    )


if __name__ == "__main__":
    DAY = sys.argv[1]
    UpdateUnifyFplay(DAY, n_cores=10)
    UnifyLocationIpFplay()
