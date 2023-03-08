
import pandas as pd
from glob import glob
import numpy as np
from datetime import datetime, timedelta
import sys

import os
import subprocess
from pyarrow import fs

from preprocessing_pgp.name.type.extractor import process_extract_name_type

os.environ['HADOOP_CONF_DIR'] = "/etc/hadoop/conf/"
os.environ['JAVA_HOME'] = "/usr/jdk64/jdk1.8.0_112"
os.environ['HADOOP_HOME'] = "/usr/hdp/3.1.0.0-78/hadoop"
os.environ['ARROW_LIBHDFS_DIR'] = "/usr/hdp/3.1.0.0-78/usr/lib/"
os.environ['CLASSPATH'] = subprocess.check_output(
    "$HADOOP_HOME/bin/hadoop classpath --glob", shell=True).decode('utf-8')
hdfs = fs.HadoopFileSystem(
    host="hdfs://hdfs-cluster.datalake.bigdata.local", port=8020)

sys.path.append('/bigdata/fdp/cdp/source/core_profile/preprocess/utils')
from preprocess_profile import (
    remove_same_username_email,
    cleansing_profile_name,
    extracting_pronoun_from_name
)
from filter_profile import get_difference_data
from const import (
    UTILS_PATH,
    CENTRALIZE_PATH,
    PREPROCESS_PATH
)

# function get profile change/new


# def DifferenceProfile(now_df, yesterday_df):
#     difference_df = now_df[~now_df.apply(tuple, 1).isin(
#         yesterday_df.apply(tuple, 1))].copy()
#     return difference_df

# function unify profile


def UnifyLongChau(
    profile_longchau: pd.DataFrame,
    n_cores: int = 1
):
    # VARIABLE
    dict_trash = {'': None, 'Nan': None, 'nan': None, 'None': None,
                  'none': None, 'Null': None, 'null': None, "''": None}

    # * Cleansing
    print(">>> Cleansing profile")
    profile_longchau = cleansing_profile_name(
        profile_longchau,
        name_col='name',
        n_cores=n_cores
    )
    profile_longchau.rename(columns={
        'email': 'email_raw',
        'phone': 'phone_raw',
        'name': 'raw_name'
    }, inplace=True)

    # * Loading dictionary
    print(">>> Loading dictionaries")
    profile_phones = profile_longchau['phone_raw'].drop_duplicates().dropna()
    profile_emails = profile_longchau['email_raw'].drop_duplicates().dropna()
    profile_names = profile_longchau['raw_name'].drop_duplicates().dropna()

    # phone, email (valid)
    valid_phone = pd.read_parquet(
        f'{UTILS_PATH}/valid_phone_latest.parquet',
        filters=[('phone_raw', 'in', profile_phones)],
        filesystem=hdfs,
        columns=['phone_raw', 'phone', 'is_phone_valid']
    )
    valid_email = pd.read_parquet(
        f'{UTILS_PATH}/valid_email_latest.parquet',
        filters=[('email_raw', 'in', profile_emails)],
        filesystem=hdfs,
        columns=['email_raw', 'email', 'is_email_valid']
    )
    dict_name_lst = pd.read_parquet(
        f'{UTILS_PATH}/dict_name_latest.parquet',
        filters=[('raw_name', 'in', profile_names)],
        filesystem=hdfs,
        columns=[
            'raw_name', 'enrich_name',
            'last_name', 'middle_name', 'first_name',
            'gender'
        ]
    ).rename(columns={
        'gender': 'gender_enrich'
    })

    # info
    print(">>> Processing Info")
    profile_longchau = profile_longchau.rename(
        columns={'customer_type': 'customer_type_longchau'})
    profile_longchau.loc[profile_longchau['gender'] == '-1', 'gender'] = None
    profile_longchau.loc[profile_longchau['address'].isin(
        ['', 'Null', 'None', 'Test']), 'address'] = None
    profile_longchau.loc[profile_longchau['address'].notna(
    ) & profile_longchau['address'].str.isnumeric(), 'address'] = None
    profile_longchau.loc[profile_longchau['address'].str.len()
                         < 5, 'address'] = None
    profile_longchau['customer_type_longchau'] =\
        profile_longchau['customer_type_longchau'].replace({
            'Individual': 'Ca nhan',
            'Company': 'Cong ty',
            'Other': None
        })

    # merge get phone, email (valid)
    print(">>> Merging phone, email, name")
    profile_longchau = pd.merge(
        profile_longchau.set_index('phone_raw'),
        valid_phone.set_index('phone_raw'),
        left_index=True, right_index=True,
        how='left',
        sort=False
    ).reset_index(drop=False)

    profile_longchau = pd.merge(
        profile_longchau.set_index('email_raw'),
        valid_email.set_index('email_raw'),
        left_index=True, right_index=True,
        how='left',
        sort=False
    ).reset_index(drop=False)

    profile_longchau = pd.merge(
        profile_longchau.set_index('raw_name'),
        dict_name_lst.set_index('raw_name'),
        left_index=True, right_index=True,
        how='left',
        sort=False
    ).rename(columns={
        'enrich_name': 'name'
    }).reset_index(drop=False)

    # Refilling info
    cant_predict_name_mask = profile_longchau['name'].isna()
    profile_longchau.loc[
        cant_predict_name_mask,
        'name'
    ] = profile_longchau.loc[
        cant_predict_name_mask,
        'raw_name'
    ]
    profile_longchau['name'] = profile_longchau['name'].replace(dict_trash)

    # customer_type
    print(">>> Processing Customer Type")
    profile_longchau = process_extract_name_type(
        profile_longchau,
        name_col='name',
        n_cores=n_cores,
        logging_info=False
    )
    profile_longchau['customer_type'] = profile_longchau['customer_type'].map({
        'customer': 'Ca nhan',
        'company': 'Cong ty',
        'medical': 'Benh vien - Phong kham',
        'edu': 'Giao duc',
        'biz': 'Ho kinh doanh'
    })
    profile_longchau.loc[
        profile_longchau['customer_type'] == 'Ca nhan',
        'customer_type'
    ] = profile_longchau['customer_type_longchau']
    profile_longchau = profile_longchau.drop(columns=['customer_type_longchau'])

    # drop name is username_email
    print(">>> Extra Cleansing Name")
    profile_longchau = remove_same_username_email(
        profile_longchau,
        name_col='name',
        email_col='email'
    )

    # clean name
    condition_name =\
        (profile_longchau['customer_type'].isin([None, 'Ca nhan', np.nan]))\
        & (profile_longchau['name'].notna())
    profile_longchau = extracting_pronoun_from_name(
        profile_longchau,
        condition=condition_name,
        name_col='name'
    )

    # is full name
    print(">>> Checking Full Name")
    profile_longchau.loc[profile_longchau['last_name'].notna(
    ) & profile_longchau['first_name'].notna(), 'is_full_name'] = True
    profile_longchau['is_full_name'] = profile_longchau['is_full_name'].fillna(
        False)
    profile_longchau = profile_longchau.drop(
        columns=['last_name', 'middle_name', 'first_name'])

    # valid gender by model
    print(">>> Validating Gender")
    profile_longchau.loc[
        profile_longchau['customer_type'] != 'Ca nhan',
        'gender'
    ] = None
    # profile_fo.loc[profile_fo['gender'].notna() & profile_fo['name'].isna(), 'gender'] = None
    profile_longchau.loc[
        (profile_longchau['gender'].notna())
        & (profile_longchau['gender'] != profile_longchau['gender_enrich']),
        'gender'
    ] = None

    # Location
#     shop_longchau = pd.read_parquet('/data/fpt/ftel/cads/dep_solution/user/namdp11/scross_fill/runner/refactor/material/longchau_shop_khanhhb3.parquet',
#                                     filesystem=hdfs, columns = ['ShopCode', 'level1_norm', 'level2_norm', 'level3_norm']).drop_duplicates()
    print(">>> Processing Address")
    path_shop = [f.path
                 for f in hdfs.get_file_info(fs.FileSelector("/data/fpt/ftel/cads/dep_solution/user/namdp11/scross_fill/runner/refactor/material/longchau_shop_khanhhb3_ver2.parquet/"))
                 ][-1]
    shop_longchau = pd.read_parquet(path_shop, filesystem=hdfs, columns=[
                                    'ShopCode', 'Level1Norm', 'Level2Norm', 'Level3Norm']).drop_duplicates()
    shop_longchau.columns = ['shop_code', 'city', 'district', 'ward']
    shop_longchau['shop_code'] = shop_longchau['shop_code'].astype(str)

    transaction_paths = sorted(
        glob('/bigdata/fdp/frt/data/posdata/pharmacy/posthuoc_ordr/*'))
    transaction_longchau = pd.DataFrame()
    for path in transaction_paths:
        df = pd.read_parquet(path)
        df = df[['CardCode', 'ShopCode', 'Source']].drop_duplicates()
        df.columns = ['cardcode', 'shop_code', 'source']
        df['shop_code'] = df['shop_code'].astype(str)

        df = pd.merge(
            df.set_index('shop_code'),
            shop_longchau.set_index('shop_code'),
            left_index=True, right_index=True,
            how='left',
            sort=False
        ).reset_index()
        df = df.sort_values(by=['cardcode', 'source'], ascending=True)
        df = df.drop_duplicates(subset=['cardcode'], keep='last')
        df = df[['cardcode', 'city', 'district',
                 'ward', 'source']].reset_index(drop=True)
        transaction_longchau = transaction_longchau.append(
            df, ignore_index=True)

    transaction_longchau = transaction_longchau.sort_values(
        by=['cardcode', 'source'], ascending=True)
    transaction_longchau = transaction_longchau.drop_duplicates(
        subset=['cardcode'], keep='last')
    transaction_longchau.to_parquet('/data/fpt/ftel/cads/dep_solution/user/namdp11/scross_fill/runner/refactor/material/longchau_location_latest_khanhhb3.parquet',
                                    filesystem=hdfs, index=False)

    # location of profile
    profile_location_longchau = pd.read_parquet('/data/fpt/ftel/cads/dep_solution/sa/cdp/data/longchau_address_latest.parquet',
                                                columns=['CardCode', 'Address', 'Ward', 'District', 'City', 'Street'], filesystem=hdfs)
    profile_location_longchau.columns = [
        'cardcode', 'address', 'ward', 'district', 'city', 'street']
    profile_location_longchau = profile_location_longchau.rename(
        columns={'cardcode': 'uid'})
    profile_location_longchau.loc[profile_location_longchau['address'].isin(
        ['', 'Null', 'None', 'Test']), 'address'] = None
    profile_location_longchau.loc[profile_location_longchau['address'].str.len(
    ) < 5, 'address'] = None
    profile_location_longchau['address'] = profile_location_longchau['address'].str.strip(
    ).replace(dict_trash)
    profile_location_longchau = profile_location_longchau.drop_duplicates(
        subset=['uid'], keep='first')

    latest_location_longchau = pd.read_parquet('/data/fpt/ftel/cads/dep_solution/user/namdp11/scross_fill/runner/refactor/material/longchau_location_latest_khanhhb3.parquet',
                                               filesystem=hdfs)
    latest_location_longchau = latest_location_longchau.drop(columns=[
                                                             'source'])
    latest_location_longchau = latest_location_longchau.rename(
        columns={'cardcode': 'uid'})
    latest_location_longchau['ward'] = None

    # source address
    latest_location_longchau.loc[latest_location_longchau['city'].notna(
    ), 'source_city'] = 'LongChau from shop'
    latest_location_longchau.loc[latest_location_longchau['district'].notna(
    ), 'source_district'] = 'LongChau from shop'
    latest_location_longchau.loc[latest_location_longchau['ward'].notna(
    ), 'source_ward'] = 'LongChau from shop'

    profile_location_longchau.loc[profile_location_longchau['city'].notna(
    ), 'source_city'] = 'LongChau from profile'
    profile_location_longchau.loc[profile_location_longchau['district'].notna(
    ), 'source_district'] = 'LongChau from profile'
    profile_location_longchau.loc[profile_location_longchau['ward'].notna(
    ), 'source_ward'] = 'LongChau from profile'

    # from shop: miss ward & district & city
    profile_location_longchau_bug = profile_location_longchau[profile_location_longchau['city'].isna() &
                                                              profile_location_longchau['district'].isna() &
                                                              profile_location_longchau['ward'].isna()]
    profile_location_longchau_bug = profile_location_longchau_bug[[
        'uid', 'address']]
    profile_location_longchau_bug = pd.merge(
        profile_location_longchau_bug.set_index('uid'),
        latest_location_longchau.set_index('uid'),
        left_index=True, right_index=True,
        how='left',
        sort=False
    ).reset_index()
    profile_location_longchau = profile_location_longchau[~profile_location_longchau['uid'].isin(
        profile_location_longchau_bug['uid'])]
    profile_location_longchau = pd.concat([
        profile_location_longchau,
        profile_location_longchau_bug
    ], ignore_index=True)

    # from shop: miss district & city
    profile_location_longchau_bug = profile_location_longchau[profile_location_longchau['city'].isna() &
                                                              profile_location_longchau['district'].isna()]
    profile_location_longchau_bug = profile_location_longchau_bug.drop(columns=['city', 'source_city',
                                                                                'district', 'source_district'])
    temp_latest_location_longchau = latest_location_longchau[['uid', 'city', 'source_city',
                                                              'district', 'source_district']]
    profile_location_longchau_bug = pd.merge(
        profile_location_longchau_bug.set_index('uid'),
        temp_latest_location_longchau.set_index('uid'),
        left_index=True, right_index=True,
        how='left',
        sort=False
    ).reset_index()
    profile_location_longchau = profile_location_longchau[~profile_location_longchau['uid'].isin(
        profile_location_longchau_bug['uid'])]
    profile_location_longchau = pd.concat([
        profile_location_longchau,
        profile_location_longchau_bug
    ], ignore_index=True)

    # from shop: miss city
    profile_location_longchau_bug = profile_location_longchau[profile_location_longchau['city'].isna() &
                                                              profile_location_longchau['district'].notna()]
    profile_location_longchau_bug = profile_location_longchau_bug.drop(
        columns=['city', 'source_city'])
    temp_latest_location_longchau = latest_location_longchau[[
        'uid', 'district', 'city', 'source_city']]
    profile_location_longchau_bug = pd.merge(
        profile_location_longchau_bug.set_index('uid'),
        temp_latest_location_longchau.set_index('uid'),
        left_index=True, right_index=True,
        how='left',
        sort=False
    ).reset_index()
    profile_location_longchau = profile_location_longchau[~profile_location_longchau['uid'].isin(
        profile_location_longchau_bug['uid'])]
    profile_location_longchau = pd.concat([
        profile_location_longchau,
        profile_location_longchau_bug
    ], ignore_index=True)

    # from shop: miss district
    profile_location_longchau_bug = profile_location_longchau[profile_location_longchau['city'].notna() &
                                                              profile_location_longchau['district'].isna()].copy()
    profile_location_longchau_bug = profile_location_longchau_bug.drop(
        columns=['district', 'source_district'])
    temp_latest_location_longchau = latest_location_longchau[[
        'uid', 'city', 'district', 'source_district']]
    profile_location_longchau_bug = pd.merge(
        profile_location_longchau_bug.set_index('uid'),
        temp_latest_location_longchau.set_index('uid'),
        left_index=True, right_index=True,
        how='left',
        sort=False
    ).reset_index()
    profile_location_longchau = profile_location_longchau[~profile_location_longchau['uid'].isin(
        profile_location_longchau_bug['uid'])]
    profile_location_longchau = pd.concat([
        profile_location_longchau,
        profile_location_longchau_bug
    ], ignore_index=True)

    # normlize address
    profile_longchau['address'] = profile_longchau['address'].str.strip().replace(
        dict_trash)
    profile_longchau = profile_longchau.drop(columns=['city'])
    profile_longchau = profile_longchau.merge(
        profile_location_longchau, how='left', on=['uid', 'address'])

    profile_longchau.loc[profile_longchau['street'].isna(), 'street'] = None
    profile_longchau.loc[profile_longchau['ward'].isna(), 'ward'] = None
    profile_longchau.loc[profile_longchau['district'].isna(),
                         'district'] = None
    profile_longchau.loc[profile_longchau['city'].isna(), 'city'] = None

    # full address
    columns = ['street', 'ward', 'district', 'city']
    profile_longchau['address'] = profile_longchau[columns].fillna('').agg(', '.join, axis=1).str.replace(
        '(?<![a-zA-Z0-9]),', '', regex=True).str.replace('-(?![a-zA-Z0-9])', '', regex=True)
    profile_longchau['address'] = profile_longchau['address'].str.strip(
        ', ').str.strip(',').str.strip()
    profile_longchau['address'] = profile_longchau['address'].str.strip().replace(
        dict_trash)
    profile_longchau.loc[profile_longchau['address'].notna(
    ), 'source_address'] = profile_longchau['source_city']

    # unit_address
    profile_longchau = profile_longchau.rename(
        columns={'street': 'unit_address'})
    profile_longchau.loc[profile_longchau['unit_address'].notna(
    ), 'source_unit_address'] = 'FSHOP from profile'

    # add info
    profile_longchau['birthday'] = None
    columns = ['uid', 'phone_raw', 'phone', 'is_phone_valid',
               'email_raw', 'email', 'is_email_valid',
               'name', 'pronoun', 'is_full_name', 'gender',
               'birthday', 'customer_type',  # 'customer_type_detail',
               'address', 'source_address', 'unit_address', 'source_unit_address',
               'ward', 'source_ward', 'district', 'source_district', 'city', 'source_city']
    profile_longchau = profile_longchau[columns]

    # Fill 'Ca nhan'
    profile_longchau.loc[profile_longchau['name'].notna(
    ) & profile_longchau['customer_type'].isna(), 'customer_type'] = 'Ca nhan'

    # return
    return profile_longchau

# function update profile (unify)


def UpdateUnifyLongChau(
    now_str:str,
    n_cores:int = 1
):
    # VARIABLES
    f_group = 'longchau'
    yesterday_str = (datetime.strptime(now_str, '%Y-%m-%d') -
                     timedelta(days=1)).strftime('%Y-%m-%d')

    # load profile (yesterday, now)
    print(">>> Loading today and yesterday profile")
    info_columns = ['uid', 'phone', 'email',
                    'name', 'gender', 'address', 'city', 'customer_type']
    now_profile = pd.read_parquet(
        f'{CENTRALIZE_PATH}/{f_group}.parquet/d={now_str}',
        filesystem=hdfs, columns=info_columns
    )
    yesterday_profile = pd.read_parquet(
        f'{CENTRALIZE_PATH}/{f_group}.parquet/d={yesterday_str}',
        filesystem=hdfs, columns=info_columns
    )

    # get profile change/new
    print(">>> Filtering new profile")
    difference_profile = get_difference_data(now_profile, yesterday_profile)
    print(f"Number of new profile {difference_profile.shape}")

    # update profile
    profile_unify = pd.read_parquet(
        f'{PREPROCESS_PATH}/{f_group}.parquet/d={yesterday_str}',
        filesystem=hdfs
    )
    if not difference_profile.empty:
        # get profile unify (old + new)
        new_profile_unify = UnifyLongChau(difference_profile, n_cores=n_cores)

        # synthetic profile
        profile_unify = pd.concat(
            [new_profile_unify, profile_unify],
            ignore_index=True
        )

    # arrange columns
    print(">>> Re-Arranging Columns")
    columns = ['uid', 'phone_raw', 'phone', 'is_phone_valid',
               'email_raw', 'email', 'is_email_valid',
               'name', 'pronoun', 'is_full_name', 'gender',
               'birthday', 'customer_type',  # 'customer_type_detail',
               'address', 'source_address', 'unit_address', 'source_unit_address',
               'ward', 'source_ward', 'district', 'source_district', 'city', 'source_city']

    profile_unify = profile_unify[columns]
    profile_unify['is_phone_valid'] =\
        profile_unify['is_phone_valid'].fillna(False)
    profile_unify['is_email_valid'] =\
        profile_unify['is_email_valid'].fillna(False)
    profile_unify = profile_unify.drop_duplicates(
        subset=['uid', 'phone_raw', 'email_raw'], keep='first')

    # save
    profile_unify['d'] = now_str
    profile_unify.to_parquet(
        f'{PREPROCESS_PATH}/{f_group}.parquet',
        filesystem=hdfs, index=False,
        partition_cols='d'
    )


if __name__ == '__main__':

    DAY = sys.argv[1]
    UpdateUnifyLongChau(DAY, n_cores=5)
