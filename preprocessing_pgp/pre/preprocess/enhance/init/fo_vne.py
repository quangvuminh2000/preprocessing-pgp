import subprocess
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from pyarrow import fs

sys.path.append("/bigdata/fdp/cdp/source/core_profile/preprocess/utils")
from const import CENTRALIZE_PATH, PREPROCESS_PATH, UTILS_PATH, hdfs


def UnifyFo(date_str: str, n_cores: int = 1):
    # VARIABLES
    f_group = "fo_vne"

    # load profile fo
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
    profile_vne = pd.read_parquet(
        f"{CENTRALIZE_PATH}/{f_group}.parquet/d={date_str}",
        filesystem=hdfs,
        columns=info_columns,
    )

    # # * Cleansing
    # print(">>> Cleansing profile")
    # profile_fo = cleansing_profile_name(
    #     profile_fo,
    #     name_col='name',
    #     n_cores=n_cores
    # )
    # profile_fo.rename(columns={
    #     'email': 'email_raw',
    #     'phone': 'phone_raw',
    #     'name': 'raw_name'
    # }, inplace=True)

    # # * Loading dictionary
    # print(">>> Loading dictionaries")
    # profile_phones = profile_fo['phone_raw'].drop_duplicates().dropna()
    # profile_emails = profile_fo['email_raw'].drop_duplicates().dropna()
    # profile_names = profile_fo['raw_name'].drop_duplicates().dropna()

    # # phone, email(valid)
    # valid_phone = pd.read_parquet(
    #     f'{UTILS_PATH}/valid_phone_latest.parquet',
    #     filters=[('phone_raw', 'in', profile_phones)],
    #     filesystem=hdfs,
    #     columns=['phone_raw', 'phone', 'is_phone_valid']
    # )
    # valid_email = pd.read_parquet(
    #     f'{UTILS_PATH}/valid_email_latest.parquet',
    #     filters=[('email_raw', 'in', profile_emails)],
    #     filesystem=hdfs,
    #     columns=['email_raw', 'email', 'is_email_valid']
    # )
    # dict_name_lst = pd.read_parquet(
    #     f'{UTILS_PATH}/dict_name_latest.parquet',
    #     filters=[('raw_name', 'in', profile_names)],
    #     filesystem=hdfs,
    #     columns=[
    #         'raw_name', 'enrich_name',
    #         'last_name', 'middle_name', 'first_name',
    #         'gender'
    #     ]
    # ).rename(columns={
    #     'gender': 'gender_enrich'
    # })

    # info
    print(">>> Processing Info")
    profile_vne = profile_vne.sort_values(by=["uid"], ascending=False)
    profile_vne = profile_vne.drop_duplicates(subset=["uid"], keep="first")

    # # merge get phone, email (valid) and names
    # print(">>> Merging phone, email, name")
    # profile_fo = pd.merge(
    #     profile_fo.set_index('phone_raw'),
    #     valid_phone.set_index('phone_raw'),
    #     left_index=True, right_index=True,
    #     how='left',
    #     sort=False
    # ).reset_index(drop=False)

    # profile_fo = pd.merge(
    #     profile_fo.set_index('email_raw'),
    #     valid_email.set_index('email_raw'),
    #     left_index=True, right_index=True,
    #     how='left',
    #     sort=False
    # ).reset_index(drop=False)

    # profile_fo = pd.merge(
    #     profile_fo.set_index('raw_name'),
    #     dict_name_lst.set_index('raw_name'),
    #     left_index=True, right_index=True,
    #     how='left',
    #     sort=False
    # ).rename(columns={
    #     'enrich_name': 'name'
    # }).reset_index(drop=False)

    # # Refilling info
    # cant_predict_name_mask =\
    #     (profile_fo['name'].isna()) & (profile_fo['raw_name'].notna())
    # profile_fo.loc[
    #     cant_predict_name_mask,
    #     'name'
    # ] = profile_fo.loc[
    #     cant_predict_name_mask,
    #     'raw_name'
    # ]
    # profile_fo['name'] = profile_fo['name'].replace(dict_trash)

    # birthday
    print(">>> Processing Birthday")
    now_year = datetime.today().year
    profile_vne.loc[profile_vne["age"] < 16, "age"] = np.nan
    profile_vne.loc[profile_vne["age"].notna(), "birthday"] = (
        (now_year - profile_vne[profile_vne["age"].notna()]["age"])
        .astype(str)
        .str.replace(".0", "", regex=False)
    )
    profile_vne = profile_vne.drop(columns=["age"])
    profile_vne.loc[profile_vne["birthday"].isna(), "birthday"] = None

    # # gender
    # print(">>> Processing Gender")
    # profile_fo['gender'] = profile_fo['gender'].replace(
    #     {'Female': 'F', 'Male': 'M', 'Other': None})

    # # customer_type
    # print(">>> Extracting customer type")
    # profile_fo = process_extract_name_type(
    #     profile_fo,
    #     name_col='name',
    #     n_cores=n_cores,
    #     logging_info=False
    # )

    # # drop name is username_email
    # print(">>> Extra Cleansing Name")
    # profile_fo = remove_same_username_email(
    #     profile_fo,
    #     name_col='name',
    #     email_col='email'
    # )

    # # clean name
    # condition_name = (profile_fo['customer_type'] == 'customer')\
    #     & (profile_fo['name'].notna())
    # profile_fo = extracting_pronoun_from_name(
    #     profile_fo,
    #     condition_name,
    #     name_col='name',
    # )

    # # is full name
    # print(">>> Checking Full Name")
    # profile_fo.loc[profile_fo['last_name'].notna(
    # ) & profile_fo['first_name'].notna(), 'is_full_name'] = True
    # profile_fo['is_full_name'] = profile_fo['is_full_name'].fillna(False)
    # profile_fo = profile_fo.drop(
    #     columns=['last_name', 'middle_name', 'first_name'])

    # # valid gender by model
    # print(">>> Validating Gender")
    # profile_fo.loc[
    #     profile_fo['customer_type'] != 'customer',
    #     'gender'
    # ] = None
    # # profile_fo.loc[profile_fo['gender'].notna() & profile_fo['name'].isna(), 'gender'] = None
    # profile_fo.loc[
    #     (profile_fo['gender'].notna())
    #     & (profile_fo['gender'] != profile_fo['gender_enrich']),
    #     'gender'
    # ] = None

    # # address, city
    # print(">>> Processing Address")
    # norm_fo_city = pd.read_parquet('/data/fpt/ftel/cads/dep_solution/user/namdp11/scross_fill/runner/refactor/material/ftel_provinces.parquet',
    #                                filesystem=hdfs)
    # norm_fo_city.columns = ['city', 'norm_city']
    # profile_fo.loc[profile_fo['address'] == 'Not set', 'address'] = None
    # profile_fo.loc[profile_fo['address'].notna(
    # ), 'city'] = profile_fo.loc[profile_fo['address'].notna(), 'address'].apply(unidecode)
    # profile_fo['city'] = profile_fo['city'].replace({'Ba Ria - Vung Tau': 'Vung Tau', 'Thua Thien Hue': 'Hue',
    #                                                  'Bac Kan': 'Bac Can', 'Dak Nong': 'Dac Nong'})
    # profile_fo = profile_fo.merge(norm_fo_city, how='left', on='city')
    # profile_fo['city'] = profile_fo['norm_city']
    # profile_fo = profile_fo.drop(columns=['norm_city'])
    # profile_fo.loc[profile_fo['city'].isna(), 'city'] = None
    # profile_fo['address'] = None

    # add info
    print(">>> Adding Temp Info")
    profile_vne[
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
    profile_vne = profile_vne[columns]
    profile_vne["birthday"] = profile_vne["birthday"].astype("datetime64[s]")
    # Fill 'Ca nhan'
    # profile_fo['customer_type'] =\
    # profile_fo['customer_type'].map({
    #     'customer': 'Ca nhan',
    #     'company': 'Cong ty',
    #     'medical': 'Benh vien - Phong kham',
    #     'edu': 'Giao duc',
    #     'biz': 'Ho kinh doanh'
    # })
    # profile_fo.loc[
    #     (profile_fo['name'].notna())
    #     & (profile_fo['customer_type'].isna()),
    #     'customer_type'
    # ] = 'Ca nhan'

    # # Fill 'Ca nhan'
    # profile_fo.loc[profile_fo['name'].notna(
    # ) & profile_fo['customer_type'].isna(), 'customer_type'] = 'Ca nhan'

    # Save
    print(f"Checking {f_group} data for {date_str}...")
    f_group_path = f"{PREPROCESS_PATH}/{f_group}.parquet"
    proc = subprocess.Popen(
        ["hdfs", "dfs", "-test", "-e", f_group_path + f"/d={date_str}"]
    )
    proc.communicate()
    if proc.returncode == 0:
        print("Data already existed, Removing...")
        subprocess.run(
            ["hdfs", "dfs", "-rm", "-r", f_group_path + f"/d={date_str}"]
        )

    profile_vne["d"] = date_str
    profile_vne.to_parquet(
        f"{PREPROCESS_PATH}/{f_group}.parquet",
        filesystem=hdfs,
        index=False,
        partition_cols="d",
        coerce_timestamps="us",
        allow_truncated_timestamps=True,
    )

    # MOST LOCATION IP
    dict_ip_path = "/data/fpt/ftel/cads/dep_solution/user/namdp11/scross_fill/runner/ip/dictionary"
    log_ip_path = "/data/fpt/ftel/cads/dep_solution/user/namdp11/scross_fill/runner/ip/fo"

    ip_cols = ["ip", "name_province", "name_district"]
    ip_location1 = pd.read_parquet(
        f"{dict_ip_path}/ip_location_batch_1.parquet", filesystem=hdfs
    )[ip_cols]
    ip_location2 = pd.read_parquet(
        f"{dict_ip_path}/ip_location_batch_2.parquet", filesystem=hdfs
    )[ip_cols]
    ip_location = pd.concat([ip_location1, ip_location2], ignore_index=True)

    # update ip
    def IpFo(date):
        date_str = date.strftime("%Y-%m-%d")
        try:
            # load log ip
            log_df = pd.read_parquet(
                f"/data/fpt/fdp/fo/dwh/stag_access_features.parquet/d={date_str}",
                filesystem=hdfs,
                columns=["user_id", "ip", "isp"],
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
            print("IP-FO Fail: {}".format(date_str))

    start_date = sorted(
        [f.path for f in hdfs.get_file_info(fs.FileSelector(log_ip_path))]
    )[-2][-18:-8]
    end_date = (datetime.today() - timedelta(days=1)).strftime("%Y-%m-%d")
    dates = pd.date_range(start_date, end_date, freq="D")

    for date in dates:
        IpFo(date)

    # stats location ip
    logs_ip_path = sorted(
        [
            f.path
            for f in hdfs.get_file_info(
                fs.FileSelector(f"{log_ip_path}/location/")
            )
        ]
    )[-180:]
    ip_fo = pd.read_parquet(logs_ip_path, filesystem=hdfs)
    stats_ip_fo = (
        ip_fo.groupby(by=["user_id", "name_province", "name_district"])["date"]
        .agg(num_date="count")
        .reset_index()
    )
    stats_ip_fo = stats_ip_fo.sort_values(
        by=["user_id", "num_date"], ascending=False
    )
    most_ip_fo = stats_ip_fo.drop_duplicates(subset=["user_id"], keep="first")
    most_ip_fo.to_parquet(
        UTILS_PATH + "/fo_location_most.parquet", index=False, filesystem=hdfs
    )


if __name__ == "__main__":
    DAY = sys.argv[1]
    UnifyFo(DAY, n_cores=5)
