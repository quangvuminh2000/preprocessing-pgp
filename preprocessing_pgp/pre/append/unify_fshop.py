
from preprocess import clean_name_cdp
import preprocess_lib
import pandas as pd
from glob import glob
import numpy as np
from unidecode import unidecode
from string import punctuation
from datetime import datetime, timedelta
from difflib import SequenceMatcher
import multiprocessing as mp
import html
import sys

import os
import subprocess
from pyarrow import fs
import pyarrow.parquet as pq

from preprocessing_pgp.name.preprocess import basic_preprocess_name
from preprocessing_pgp.name.split_name import NameProcess

os.environ['HADOOP_CONF_DIR'] = "/etc/hadoop/conf/"
os.environ['JAVA_HOME'] = "/usr/jdk64/jdk1.8.0_112"
os.environ['HADOOP_HOME'] = "/usr/hdp/3.1.0.0-78/hadoop"
os.environ['ARROW_LIBHDFS_DIR'] = "/usr/hdp/3.1.0.0-78/usr/lib/"
os.environ['CLASSPATH'] = subprocess.check_output(
    "$HADOOP_HOME/bin/hadoop classpath --glob", shell=True).decode('utf-8')
hdfs = fs.HadoopFileSystem(
    host="hdfs://hdfs-cluster.datalake.bigdata.local", port=8020)

sys.path.append('/bigdata/fdp/cdp/cdp_pages/scripts_hdfs/pre/utils/')

sys.path.append(
    '/bigdata/fdp/cdp/cdp_pages/scripts_hdfs/pre/utils/fill_accent_name/scripts')

ROOT_PATH = '/data/fpt/ftel/cads/dep_solution/sa/cdp/core'

# function get profile change/new


def DifferenceProfile(now_df, yesterday_df):
    difference_df = now_df[~now_df.apply(tuple, 1).isin(
        yesterday_df.apply(tuple, 1))].copy()
    return difference_df

# function unify profile


def UnifyFshop(profile_fshop: pd.DataFrame):
    # VARIABLE
    dict_trash = {'': None, 'Nan': None, 'nan': None, 'None': None,
                  'none': None, 'Null': None, 'null': None, "''": None}

    # * Cleansing
    print(">>> Cleansing profile")
    condition_name = profile_fshop['name'].notna()
    profile_fshop.loc[condition_name, 'name'] =\
        profile_fshop.loc[condition_name, 'name']\
        .apply(basic_preprocess_name)
    profile_fshop.rename(columns={
        'email': 'email_raw',
        'phone': 'phone_raw',
        'name': 'raw_name'
    }, inplace=True)

    # * Loadding dictionary
    print(">>> Loading dictionaries")
    profile_phones = profile_fshop['phone_raw'].drop_duplicates().dropna()
    profile_emails = profile_fshop['email_raw'].drop_duplicates().dropna()
    profile_names = profile_fshop['raw_name'].drop_duplicates().dropna()

    # phone, email (valid)
    valid_phone = pd.read_parquet(
        f'{ROOT_PATH}/utils/valid_phone_latest.parquet',
        filters=[('phone_raw', 'in', profile_phones)],
        filesystem=hdfs,
        columns=['phone_raw', 'phone', 'is_phone_valid']
    )
    valid_email = pd.read_parquet(
        f'{ROOT_PATH}/utils/valid_email_latest.parquet',
        filters=[('email_raw', 'in', profile_emails)],
        filesystem=hdfs,
        columns=['email_raw', 'email', 'is_email_valid']
    )
    dict_name_lst = pd.read_parquet(
        f'{ROOT_PATH}/utils/dict_name_latest_new.parquet',
        filters=[('raw_name', 'in', profile_names)],
        filesystem=hdfs,
        columns=[
            'raw_name', 'enrich_name',
            'last_name', 'middle_name', 'first_name',
            'gender', 'customer_type'
        ]
    ).rename(columns={
        'gender': 'gender_enrich'
    })

    # info
    print(">>> Processing Info")
    profile_fshop = profile_fshop.rename(
        columns={'customer_type': 'customer_type_fshop'})
    profile_fshop.loc[profile_fshop['gender'] == '-1', 'gender'] = None
    profile_fshop.loc[profile_fshop['address'].isin(
        ['', 'Null', 'None', 'Test']), 'address'] = None
    profile_fshop.loc[profile_fshop['address'].notna(
    ) & profile_fshop['address'].str.isnumeric(), 'address'] = None
    profile_fshop.loc[profile_fshop['address'].str.len() < 5, 'address'] = None
    profile_fshop['customer_type_fshop'] =\
        profile_fshop['customer_type_fshop'].replace({
            'Individual': 'Ca nhan',
            'Company': 'Cong ty',
            'Other': None
        })

    # merge get phone, email (valid)
    print(">>> Merging phone, email, name")
    profile_fshop = pd.merge(
        profile_fshop.set_index('phone_raw'),
        valid_phone.set_index('phone_raw'),
        left_index=True, right_index=True,
        how='left',
        sort=False
    ).reset_index(drop=False)

    profile_fshop = pd.merge(
        profile_fshop.set_index('email_raw'),
        valid_email.set_index('email_raw'),
        left_index=True, right_index=True,
        how='left',
        sort=False
    ).reset_index(drop=False)

    profile_fshop = pd.merge(
        profile_fshop.set_index('raw_name'),
        dict_name_lst.set_index('raw_name'),
        left_index=True, right_index=True,
        how='left',
        sort=False
    ).rename(columns={
        'enrich_name': 'name'
    }).reset_index(drop=False)

    # customer_type
    print(">>> Processing Customer Type")
    profile_fshop['customer_type'] = profile_fshop['customer_type'].map({
        'customer': 'Ca nhan',
        'company': 'Cong ty',
        'medical': 'Benh vien - Phong kham',
        'edu': 'Giao duc',
        'biz': 'Ho kinh doanh'
    })

    profile_fshop.loc[
        profile_fshop['customer_type'] == 'Ca nhan',
        'customer_type'
    ] = profile_fshop['customer_type_fshop']
    profile_fshop = profile_fshop.drop(columns=['customer_type_fshop'])

    # drop name is username_email
    print(">>> Extra Cleansing Name")
    profile_fshop['username_email'] = profile_fshop['email'].str.split(
        '@').str[0]
    profile_fshop.loc[profile_fshop['name'] ==
                      profile_fshop['username_email'], 'name'] = None
    profile_fshop = profile_fshop.drop(columns=['username_email'])

    # clean name
    name_process = NameProcess()
    condition_name =\
        (profile_fshop['customer_type'].isin([None, 'Ca nhan', np.nan]))\
        & (profile_fshop['name'].notna())
    profile_fshop.loc[
        condition_name,
        ['clean_name', 'pronoun']
    ] = profile_fshop.loc[condition_name, 'name']\
        .apply(name_process.CleanName).tolist()

    profile_fshop.loc[
        profile_fshop['customer_type'].isin([None, 'Ca nhan', np.nan]),
        'name'
    ] = profile_fshop['clean_name']
    profile_fshop = profile_fshop.drop(columns=['clean_name'])

    # skip pronoun
    profile_fshop['name'] = profile_fshop['name'].str.strip().str.title()
    skip_names = ['Vợ', 'Vo', 'Anh', 'Chị', 'Chi', 'Mẹ', 'Me', 'Em', 'Ba',
                  'Chú', 'Chu', 'Bác', 'Bac', 'Ông', 'Ong', 'Cô', 'Co', 'Cha', 'Dì', 'Dượng']
    profile_fshop.loc[profile_fshop['name'].isin(skip_names), 'name'] = None

    # is full name
    print(">>> Checking Full Name")
    profile_fshop.loc[profile_fshop['last_name'].notna(
    ) & profile_fshop['first_name'].notna(), 'is_full_name'] = True
    profile_fshop['is_full_name'] = profile_fshop['is_full_name'].fillna(False)
    profile_fshop = profile_fshop.drop(
        columns=['last_name', 'middle_name', 'first_name'])
    profile_fshop['name'] = profile_fshop['name'].str.strip().str.title()

    # valid gender by model
    print(">>> Validating Gender")
    profile_fshop.loc[
        profile_fshop['customer_type'] != 'Ca nhan',
        'gender'
    ] = None
    # profile_fo.loc[profile_fo['gender'].notna() & profile_fo['name'].isna(), 'gender'] = None
    profile_fshop.loc[
        (profile_fshop['gender'].notna())
        & (profile_fshop['gender'] != profile_fshop['gender_enrich']),
        'gender'
    ] = None

    # location of shop
#     shop_fshop = pd.read_parquet('/data/fpt/ftel/cads/dep_solution/user/namdp11/scross_fill/runner/refactor/material/fshop_shop_khanhhb3.parquet',
#                                  filesystem=hdfs, columns = ['ShopCode', 'LV1_NORM', 'LV2_NORM', 'LV3_NORM']).drop_duplicates()

    print(">>> Processing Address")
    path_shop = [f.path
                 for f in hdfs.get_file_info(fs.FileSelector("/data/fpt/ftel/cads/dep_solution/user/namdp11/scross_fill/runner/refactor/material/fshop_shop_khanhhb3_ver2.parquet/"))
                 ][-1]
    shop_fshop = pd.read_parquet(path_shop, filesystem=hdfs, columns=[
                                 'ShopCode', 'Level1Norm', 'Level2Norm', 'Level3Norm']).drop_duplicates()
    shop_fshop.columns = ['shop_code', 'city', 'district', 'ward']
    shop_fshop['shop_code'] = shop_fshop['shop_code'].astype(str)

    transaction_paths = sorted(
        glob('/bigdata/fdp/frt/data/posdata/ict/pos_ordr/*'))
    transaction_fshop = pd.DataFrame()
    for path in transaction_paths:
        df = pd.read_parquet(path)
        df = df[['CardCode', 'ShopCode', 'Source']].drop_duplicates()
        df.columns = ['cardcode', 'shop_code', 'source']
        df['shop_code'] = df['shop_code'].astype(str)

        df = pd.merge(
            df.set_index('shop_code'),
            shop_fshop.set_index('shop_code'),
            left_index=True, right_index=True,
            how='left',
            sort=False
        ).reset_index()
        df = df.sort_values(by=['cardcode', 'source'], ascending=True)
        df = df.drop_duplicates(subset=['cardcode'], keep='last')
        df = df[['cardcode', 'city', 'district',
                 'ward', 'source']].reset_index(drop=True)
        transaction_fshop = pd.concat([
            transaction_fshop,
            df
        ], ignore_index=True)

    transaction_fshop = transaction_fshop.sort_values(
        by=['cardcode', 'source'], ascending=True)
    transaction_fshop = transaction_fshop.drop_duplicates(
        subset=['cardcode'], keep='last')
    transaction_fshop.to_parquet('/data/fpt/ftel/cads/dep_solution/user/namdp11/scross_fill/runner/refactor/material/fshop_location_latest_khanhhb3.parquet',
                                 index=False, filesystem=hdfs)

    # location of profile
    profile_location_fshop = pd.read_parquet('/data/fpt/ftel/cads/dep_solution/sa/cdp/data/fshop_address_latest.parquet',
                                             columns=['CardCode', 'Address', 'Ward', 'District', 'City', 'Street'], filesystem=hdfs)
    profile_location_fshop.columns = [
        'cardcode', 'address', 'ward', 'district', 'city', 'street']
    profile_location_fshop = profile_location_fshop.rename(
        columns={'cardcode': 'cardcode_fshop'})
    profile_location_fshop.loc[profile_location_fshop['address'].isin(
        ['', 'Null', 'None', 'Test']), 'address'] = None
    profile_location_fshop.loc[profile_location_fshop['address'].str.len(
    ) < 5, 'address'] = None
    profile_location_fshop['address'] = profile_location_fshop['address'].str.strip(
    ).replace(dict_trash)
    profile_location_fshop = profile_location_fshop.drop_duplicates(
        subset=['cardcode_fshop'], keep='first')

    latest_location_fshop = pd.read_parquet('/data/fpt/ftel/cads/dep_solution/user/namdp11/scross_fill/runner/refactor/material/fshop_location_latest_khanhhb3.parquet',
                                            filesystem=hdfs)
    latest_location_fshop = latest_location_fshop.drop(columns=['source'])
    latest_location_fshop = latest_location_fshop.rename(
        columns={'cardcode': 'cardcode_fshop'})
    latest_location_fshop['ward'] = None

    # source address
    latest_location_fshop.loc[latest_location_fshop['city'].notna(
    ), 'source_city'] = 'FSHOP from shop'
    latest_location_fshop.loc[latest_location_fshop['district'].notna(
    ), 'source_district'] = 'FSHOP from shop'
    latest_location_fshop.loc[latest_location_fshop['ward'].notna(
    ), 'source_ward'] = 'FSHOP from shop'

    profile_location_fshop.loc[profile_location_fshop['city'].notna(
    ), 'source_city'] = 'FSHOP from profile'
    profile_location_fshop.loc[profile_location_fshop['district'].notna(
    ), 'source_district'] = 'FSHOP from profile'
    profile_location_fshop.loc[profile_location_fshop['ward'].notna(
    ), 'source_ward'] = 'FSHOP from profile'

    # from shop: miss ward & district & city
    profile_location_fshop_bug = profile_location_fshop[
        (profile_location_fshop['city'].isna())
        & (profile_location_fshop['district'].isna())
        & (profile_location_fshop['ward'].isna())
    ]
    profile_location_fshop_bug =\
        profile_location_fshop_bug[['cardcode_fshop', 'address']]
    profile_location_fshop_bug = pd.merge(
        profile_location_fshop_bug.set_index('cardcode_fshop'),
        latest_location_fshop.set_index('cardcode_fshop'),
        left_index=True, right_index=True,
        how='left',
        sort=False
    ).reset_index()
    profile_location_fshop =\
        profile_location_fshop[
            ~profile_location_fshop['cardcode_fshop'].isin(
                profile_location_fshop_bug['cardcode_fshop'])
        ]
    profile_location_fshop = pd.concat([
        profile_location_fshop,
        profile_location_fshop_bug
    ], ignore_index=True)

    # from shop: miss district & city
    profile_location_fshop_bug = profile_location_fshop[profile_location_fshop['city'].isna() &
                                                        profile_location_fshop['district'].isna()]
    profile_location_fshop_bug = profile_location_fshop_bug.drop(columns=['city', 'source_city',
                                                                          'district', 'source_district'])
    temp_latest_location_fshop = latest_location_fshop[['cardcode_fshop', 'city', 'source_city',
                                                        'district', 'source_district']]
    profile_location_fshop_bug = pd.merge(
        profile_location_fshop_bug.set_index('cardcode_fshop'),
        temp_latest_location_fshop.set_index('cardcode_fshop'),
        left_index=True, right_index=True,
        how='left',
        sort=False
    ).reset_index()
    profile_location_fshop = profile_location_fshop[~profile_location_fshop['cardcode_fshop'].isin(
        profile_location_fshop_bug['cardcode_fshop'])]
    profile_location_fshop = pd.concat([
        profile_location_fshop,
        profile_location_fshop_bug
    ], ignore_index=True)

    # from shop: miss city
    profile_location_fshop_bug = profile_location_fshop[profile_location_fshop['city'].isna() &
                                                        profile_location_fshop['district'].notna()]
    profile_location_fshop_bug = profile_location_fshop_bug.drop(
        columns=['city', 'source_city'])
    temp_latest_location_fshop = latest_location_fshop[[
        'cardcode_fshop', 'district', 'city', 'source_city']]
    profile_location_fshop_bug = pd.merge(
        profile_location_fshop_bug.set_index('cardcode_fshop'),
        temp_latest_location_fshop.set_index('cardcode_fshop'),
        left_index=True, right_index=True,
        how='left',
        sort=False
    ).reset_index()
    profile_location_fshop = profile_location_fshop[~profile_location_fshop['cardcode_fshop'].isin(
        profile_location_fshop_bug['cardcode_fshop'])]
    profile_location_fshop = pd.concat([
        profile_location_fshop,
        profile_location_fshop_bug
    ], ignore_index=True)

    # from shop: miss district
    profile_location_fshop_bug = profile_location_fshop[profile_location_fshop['city'].notna() &
                                                        profile_location_fshop['district'].isna()]
    profile_location_fshop_bug = profile_location_fshop_bug.drop(
        columns=['district', 'source_district'])
    temp_latest_location_fshop = latest_location_fshop[[
        'cardcode_fshop', 'city', 'district', 'source_district']]
    profile_location_fshop_bug = pd.merge(
        profile_location_fshop_bug.set_index('cardcode_fshop'),
        temp_latest_location_fshop.set_index('cardcode_fshop'),
        left_index=True, right_index=True,
        how='left',
        sort=False
    ).reset_index()
    profile_location_fshop = profile_location_fshop[~profile_location_fshop['cardcode_fshop'].isin(
        profile_location_fshop_bug['cardcode_fshop'])]
    profile_location_fshop = pd.concat([
        profile_location_fshop,
        profile_location_fshop_bug
    ], ignore_index=True)

    # normlize address
    profile_fshop['address'] = profile_fshop['address'].str.strip().replace(
        dict_trash)
    profile_fshop = profile_fshop.drop(columns=['city'])
    profile_fshop = profile_fshop.merge(profile_location_fshop, how='left', on=[
                                        'cardcode_fshop', 'address'])

    profile_fshop.loc[profile_fshop['street'].isna(), 'street'] = None
    profile_fshop.loc[profile_fshop['ward'].isna(), 'ward'] = None
    profile_fshop.loc[profile_fshop['district'].isna(), 'district'] = None
    profile_fshop.loc[profile_fshop['city'].isna(), 'city'] = None

    # full address
    columns = ['street', 'ward', 'district', 'city']
    profile_fshop['address'] = profile_fshop[columns].fillna('').agg(', '.join, axis=1).str.replace(
        '(?<![a-zA-Z0-9]),', '', regex=True).str.replace('-(?![a-zA-Z0-9])', '', regex=True)
    profile_fshop['address'] = profile_fshop['address'].str.strip(
        ', ').str.strip(',').str.strip()
    profile_fshop['address'] = profile_fshop['address'].str.strip().replace(
        dict_trash)
    profile_fshop.loc[profile_fshop['address'].notna(
    ), 'source_address'] = profile_fshop['source_city']

    # unit_address
    profile_fshop = profile_fshop.rename(columns={'street': 'unit_address'})
    profile_fshop.loc[profile_fshop['unit_address'].notna(
    ), 'source_unit_address'] = 'FSHOP from profile'

    # add info
    print(">>> Adding Temp Info")
    profile_fshop['birthday'] = None
    columns = ['cardcode_fshop', 'phone_raw', 'phone', 'is_phone_valid',
               'email_raw', 'email', 'is_email_valid',
               'name', 'pronoun', 'is_full_name', 'gender',
               'birthday', 'customer_type',  # 'customer_type_detail',
               'address', 'source_address', 'unit_address', 'source_unit_address',
               'ward', 'source_ward', 'district', 'source_district', 'city', 'source_city']
    profile_fshop = profile_fshop[columns]

    # Fill 'Ca nhan'
    profile_fshop.loc[
        (profile_fshop['name'].notna())
        & (profile_fshop['customer_type'].isna()),
        'customer_type'
    ] = 'Ca nhan'

    # return
    return profile_fshop

# function update profile (unify)


def UpdateUnifyFshop(now_str):
    # VARIABLES
    raw_path = ROOT_PATH + '/raw'
    unify_path = ROOT_PATH + '/pre'
    f_group = 'fshop'
    yesterday_str = (datetime.strptime(now_str, '%Y-%m-%d') -
                     timedelta(days=1)).strftime('%Y-%m-%d')

    # load profile (yesterday, now)
    print(">>> Loading today and yesterday profile")
    info_columns = ['cardcode_fshop', 'phone', 'email',
                    'name', 'gender', 'address', 'city', 'customer_type']
    now_profile = pd.read_parquet(
        f'{raw_path}/{f_group}.parquet/d={now_str}',
        filesystem=hdfs, columns=info_columns
    )
    yesterday_profile = pd.read_parquet(
        f'{raw_path}/{f_group}.parquet/d={yesterday_str}',
        filesystem=hdfs, columns=info_columns
    )

    # get profile change/new
    print(">>> Filtering new profile")
    difference_profile = DifferenceProfile(now_profile, yesterday_profile)

    # update profile
    profile_unify = pd.read_parquet(
        f'{unify_path}/{f_group}.parquet/d={yesterday_str}',
        filesystem=hdfs
    )
    if not difference_profile.empty:
        # get profile unify (old + new)
        new_profile_unify = UnifyFshop(difference_profile)

        # synthetic profile
        profile_unify = pd.concat(
            [new_profile_unify, profile_unify],
            ignore_index=True
        )

    # arrange columns
    print(">>> Re-Arranging Columns")
    columns = ['cardcode_fshop', 'phone_raw', 'phone', 'is_phone_valid',
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
        subset=['cardcode_fshop', 'phone_raw', 'email_raw'],
        keep='first'
    )

    # save
    profile_unify['d'] = now_str
    profile_unify.to_parquet(
        f'{unify_path}/{f_group}.parquet',
        filesystem=hdfs, index=False,
        partition_cols='d'
    )


if __name__ == '__main__':

    now_str = sys.argv[1]
    UpdateUnifyFshop(now_str)
