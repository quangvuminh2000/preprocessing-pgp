
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
os.environ['HADOOP_CONF_DIR'] = "/etc/hadoop/conf/"
os.environ['JAVA_HOME'] = "/usr/jdk64/jdk1.8.0_112"
os.environ['HADOOP_HOME'] = "/usr/hdp/3.1.0.0-78/hadoop"
os.environ['ARROW_LIBHDFS_DIR'] = "/usr/hdp/3.1.0.0-78/usr/lib/"
os.environ['CLASSPATH'] = subprocess.check_output("$HADOOP_HOME/bin/hadoop classpath --glob", shell=True).decode('utf-8')
hdfs = fs.HadoopFileSystem(host="hdfs://hdfs-cluster.datalake.bigdata.local", port=8020)

sys.path.append('/bigdata/fdp/cdp/cdp_pages/scripts_hdfs/pre/utils/')
import preprocess_lib

sys.path.append('/bigdata/fdp/cdp/cdp_pages/scripts_hdfs/pre/utils/fill_accent_name/scripts')
from preprocess import clean_name_cdp

def UnifyLongChau(date_str):
    # VARIABLES
    raw_path = '/data/fpt/ftel/cads/dep_solution/sa/dev/raw'
    unify_path = '/data/fpt/ftel/cads/dep_solution/sa/dev/pre'
    f_group = 'longchau'

    dict_trash = {'': None, 'Nan': None, 'nan': None, 'None': None, 'none': None, 'Null': None, 'null': None, "''": None}
    dict_location = pd.read_parquet('/data/fpt/ftel/cads/dep_solution/user/namdp11/scross_fill/runner/refactor/material/dict_location.parquet', 
                                    filesystem=hdfs)

    # phone, email (valid)
    valid_phone = pd.read_parquet('/data/fpt/ftel/cads/dep_solution/sa/dev/utils/valid_phone_latest.parquet',
                                  filesystem=hdfs, columns=['phone_raw', 'phone', 'is_phone_valid'])
    valid_email = pd.read_parquet('/data/fpt/ftel/cads/dep_solution/sa/dev/utils/valid_email_latest.parquet',
                                  filesystem=hdfs, columns=['email_raw', 'email', 'is_email_valid'])

    # load profile longchau
    profile_longchau = pd.read_parquet(f'{raw_path}/{f_group}.parquet/d={date_str}', filesystem=hdfs)

    # info
    profile_longchau = profile_longchau[['cardcode_longchau', 'phone', 'email', 'name', 'gender', 'address', 'city', 'customer_type',]].copy()
    profile_longchau = profile_longchau.rename(columns={'customer_type': 'customer_type_longchau'})
    profile_longchau.loc[profile_longchau['gender'] == '-1', 'gender'] = None
    profile_longchau.loc[profile_longchau['address'].isin(['', 'Null', 'None', 'Test']), 'address'] = None
    profile_longchau.loc[profile_longchau['address'].notna() & profile_longchau['address'].str.isnumeric(), 'address'] = None
    profile_longchau.loc[profile_longchau['address'].str.len() < 5, 'address'] = None
    profile_longchau['customer_type_longchau'] = profile_longchau['customer_type_longchau'].replace({'Individual': 'Ca nhan', 
                                                                                                     'Company': 'Cong ty', 
                                                                                                     'Other': None})

    # merge get phone, email (valid)
    profile_longchau = profile_longchau.rename(columns={'phone': 'phone_raw', 'email': 'email_raw'})
    profile_longchau = profile_longchau.merge(valid_phone, how='left', on=['phone_raw'])
    profile_longchau = profile_longchau.merge(valid_email, how='left', on=['email_raw'])
    
    # customer_type
    condition_name = profile_longchau['name'].notna()
    profile_longchau.loc[condition_name, 'name'] = profile_longchau.loc[condition_name, 'name'].apply(clean_name_cdp)
    
    name_longchau = profile_longchau[profile_longchau['name'].notna()][['name']].copy().drop_duplicates()
    name_longchau = preprocess_lib.ExtractCustomerType(name_longchau)
    profile_longchau = profile_longchau.merge(name_longchau, how='left', on=['name'])
    profile_longchau.loc[profile_longchau['customer_type'].isna(), 'customer_type'] = profile_longchau['customer_type_longchau']
    profile_longchau = profile_longchau.drop(columns=['customer_type_longchau'])
    profile_longchau.loc[profile_longchau['customer_type'].isna(), 'customer_type'] = None
#     profile_longchau.loc[profile_longchau['customer_type_detail'].isna(), 'customer_type_detail'] = None

    # drop name is username_email
    profile_longchau['username_email'] = profile_longchau['email'].str.split('@').str[0]
    profile_longchau.loc[profile_longchau['name'] == profile_longchau['username_email'], 'name'] = None
    profile_longchau = profile_longchau.drop(columns=['username_email'])

    # clean name
    condition_name = profile_longchau['customer_type'].isin([None, 'Ca nhan', np.nan]) & profile_longchau['name'].notna()
    profile_longchau.loc[
        condition_name,
        ['clean_name', 'pronoun']
    ] = profile_longchau.loc[condition_name, ].apply(lambda row: preprocess_lib.CleanName(row['name'], row['email']), axis=1).tolist()
    profile_longchau.loc[profile_longchau['customer_type'].isin([None, 'Ca nhan', np.nan]), 'name'] = profile_longchau['clean_name']
    profile_longchau = profile_longchau.drop(columns=['clean_name'])
    
    # skip pronoun
    profile_longchau['name'] = profile_longchau['name'].str.strip().str.title()
    skip_names = ['Vợ', 'Vo', 'Anh', 'Chị', 'Chi', 'Mẹ', 'Me', 'Em', 'Ba', 'Chú', 'Chu', 'Bác', 'Bac', 'Ông', 'Ong', 'Cô', 'Co', 'Cha', 'Dì', 'Dượng']
    profile_longchau.loc[profile_longchau['name'].isin(skip_names), 'name'] = None

    # format name
    with mp.Pool(8) as pool:
        profile_longchau[['last_name', 'middle_name', 'first_name']] = pool.map(preprocess_lib.SplitName, profile_longchau['name'])
    columns = ['last_name', 'middle_name', 'first_name']
    profile_longchau.loc[condition_name, 'name'] = profile_longchau[columns].fillna('').agg(' '.join, axis=1).str.replace('(?<![a-zA-Z0-9]),', '', regex=True).str.replace('-(?![a-zA-Z0-9])', '', regex=True)
    profile_longchau['name'] = profile_longchau['name'].str.strip().replace(dict_trash)
    profile_longchau.loc[profile_longchau['name'].isna(), 'name'] = None

    # is full name
    profile_longchau.loc[profile_longchau['last_name'].notna() & profile_longchau['first_name'].notna(), 'is_full_name'] = True
    profile_longchau['is_full_name'] = profile_longchau['is_full_name'].fillna(False)
    profile_longchau = profile_longchau.drop(columns=['last_name', 'middle_name', 'first_name'])
    profile_longchau['name'] = profile_longchau['name'].str.strip().str.title()

    # valid gender by model
    profile_longchau.loc[profile_longchau['customer_type'] != 'Ca nhan', 'gender'] = None
    profile_longchau_1 = profile_longchau[profile_longchau['name'].notna()].copy()
    profile_longchau_2 = profile_longchau[profile_longchau['name'].isna()].copy()
    profile_longchau_1 = preprocess_lib.Name2Gender(profile_longchau_1, name_col='name')
    profile_longchau_1.loc[profile_longchau_1['gender'].notna() & 
                           (profile_longchau_1['gender'] != profile_longchau_1['predict_gender']), 'gender'] = None
    profile_longchau_1 = profile_longchau_1.drop(columns=['predict_gender'])
    profile_longchau_2['gender'] = None
    profile_longchau = pd.concat([profile_longchau_1, profile_longchau_2], ignore_index=True).sample(frac=1)

    # Location
#     shop_longchau = pd.read_parquet('/data/fpt/ftel/cads/dep_solution/user/namdp11/scross_fill/runner/refactor/material/longchau_shop_khanhhb3.parquet', 
#                                     filesystem=hdfs, columns = ['ShopCode', 'level1_norm', 'level2_norm', 'level3_norm']).drop_duplicates()

    
    path_shop = [f.path
                 for f in hdfs.get_file_info(fs.FileSelector("/data/fpt/ftel/cads/dep_solution/user/namdp11/scross_fill/runner/refactor/material/longchau_shop_khanhhb3_ver2.parquet/"))
                ][-1]
    shop_longchau = pd.read_parquet(path_shop, filesystem=hdfs, columns = ['ShopCode', 'Level1Norm', 'Level2Norm', 'Level3Norm']).drop_duplicates()
    shop_longchau.columns = ['shop_code', 'city', 'district', 'ward']
    shop_longchau['shop_code'] = shop_longchau['shop_code'].astype(str)

    transaction_paths = sorted(glob('/bigdata/fdp/frt/data/posdata/pharmacy/posthuoc_ordr/*'))
    transaction_longchau = pd.DataFrame()
    for path in transaction_paths:
        df = pd.read_parquet(path)
        df = df[['CardCode', 'ShopCode', 'Source']].drop_duplicates()
        df.columns = ['cardcode', 'shop_code', 'source']
        df['shop_code'] = df['shop_code'].astype(str)

        df = df.merge(shop_longchau, how='left', on='shop_code')
        df = df.sort_values(by=['cardcode', 'source'], ascending=True)
        df = df.drop_duplicates(subset=['cardcode'], keep='last')
        df = df[['cardcode', 'city', 'district', 'ward', 'source']].reset_index(drop=True)
        transaction_longchau = transaction_longchau.append(df, ignore_index=True)

    transaction_longchau = transaction_longchau.sort_values(by=['cardcode', 'source'], ascending=True)
    transaction_longchau = transaction_longchau.drop_duplicates(subset=['cardcode'], keep='last')
    transaction_longchau.to_parquet('/data/fpt/ftel/cads/dep_solution/user/namdp11/scross_fill/runner/refactor/material/longchau_location_latest_khanhhb3.parquet', 
                                    filesystem=hdfs, index=False) 

    # location of profile
    profile_location_longchau = pd.read_parquet('/data/fpt/ftel/cads/dep_solution/sa/cdp/data/longchau_address_latest.parquet', 
                                             columns=['CardCode', 'Address', 'Ward', 'District', 'City', 'Street'], filesystem=hdfs)
    profile_location_longchau.columns = ['cardcode', 'address', 'ward', 'district', 'city', 'street']
    profile_location_longchau = profile_location_longchau.rename(columns={'cardcode': 'cardcode_longchau'})
    profile_location_longchau.loc[profile_location_longchau['address'].isin(['', 'Null', 'None', 'Test']), 'address'] = None
    profile_location_longchau.loc[profile_location_longchau['address'].str.len() < 5, 'address'] = None
    profile_location_longchau['address'] = profile_location_longchau['address'].str.strip().replace(dict_trash)
    profile_location_longchau = profile_location_longchau.drop_duplicates(subset=['cardcode_longchau'], keep='first')

    latest_location_longchau = pd.read_parquet('/data/fpt/ftel/cads/dep_solution/user/namdp11/scross_fill/runner/refactor/material/longchau_location_latest_khanhhb3.parquet', 
                                               filesystem=hdfs)
    latest_location_longchau = latest_location_longchau.drop(columns=['source'])
    latest_location_longchau = latest_location_longchau.rename(columns={'cardcode': 'cardcode_longchau'})
    latest_location_longchau['ward'] = None

    # source address
    latest_location_longchau.loc[latest_location_longchau['city'].notna(), 'source_city'] = 'LongChau from shop'
    latest_location_longchau.loc[latest_location_longchau['district'].notna(), 'source_district'] = 'LongChau from shop'
    latest_location_longchau.loc[latest_location_longchau['ward'].notna(), 'source_ward'] = 'LongChau from shop'

    profile_location_longchau.loc[profile_location_longchau['city'].notna(), 'source_city'] = 'LongChau from profile'
    profile_location_longchau.loc[profile_location_longchau['district'].notna(), 'source_district'] = 'LongChau from profile'
    profile_location_longchau.loc[profile_location_longchau['ward'].notna(), 'source_ward'] = 'LongChau from profile'

    ## from shop: miss ward & district & city
    profile_location_longchau_bug = profile_location_longchau[profile_location_longchau['city'].isna() & 
                                                              profile_location_longchau['district'].isna() & 
                                                              profile_location_longchau['ward'].isna()].copy()
    profile_location_longchau_bug = profile_location_longchau_bug[['cardcode_longchau', 'address']].copy()
    profile_location_longchau_bug = profile_location_longchau_bug.merge(latest_location_longchau, how='left',
                                                                        on=['cardcode_longchau'])
    profile_location_longchau = profile_location_longchau[~profile_location_longchau['cardcode_longchau'].isin(profile_location_longchau_bug['cardcode_longchau'])]
    profile_location_longchau = profile_location_longchau.append(profile_location_longchau_bug, 
                                                                 ignore_index=True)

    ## from shop: miss district & city
    profile_location_longchau_bug = profile_location_longchau[profile_location_longchau['city'].isna() & 
                                                              profile_location_longchau['district'].isna()].copy()
    profile_location_longchau_bug = profile_location_longchau_bug.drop(columns=['city', 'source_city',
                                                                                'district', 'source_district'])
    temp_latest_location_longchau = latest_location_longchau[['cardcode_longchau', 'city', 'source_city',
                                                              'district', 'source_district']]
    profile_location_longchau_bug = profile_location_longchau_bug.merge(temp_latest_location_longchau, how='left', 
                                                                        on=['cardcode_longchau'])
    profile_location_longchau = profile_location_longchau[~profile_location_longchau['cardcode_longchau'].isin(profile_location_longchau_bug['cardcode_longchau'])]
    profile_location_longchau = profile_location_longchau.append(profile_location_longchau_bug, 
                                                                 ignore_index=True)

    ## from shop: miss city
    profile_location_longchau_bug = profile_location_longchau[profile_location_longchau['city'].isna() & 
                                                              profile_location_longchau['district'].notna()].copy()
    profile_location_longchau_bug = profile_location_longchau_bug.drop(columns=['city', 'source_city'])
    temp_latest_location_longchau = latest_location_longchau[['cardcode_longchau', 'district', 'city', 'source_city']]
    profile_location_longchau_bug = profile_location_longchau_bug.merge(temp_latest_location_longchau, how='left', 
                                                                        on=['cardcode_longchau', 'district'])
    profile_location_longchau = profile_location_longchau[~profile_location_longchau['cardcode_longchau'].isin(profile_location_longchau_bug['cardcode_longchau'])]
    profile_location_longchau = profile_location_longchau.append(profile_location_longchau_bug, 
                                                                 ignore_index=True)

    # from shop: miss district
    profile_location_longchau_bug = profile_location_longchau[profile_location_longchau['city'].notna() & 
                                                              profile_location_longchau['district'].isna()].copy()
    profile_location_longchau_bug = profile_location_longchau_bug.drop(columns=['district', 'source_district'])
    temp_latest_location_longchau = latest_location_longchau[['cardcode_longchau', 'city', 'district', 'source_district']]
    profile_location_longchau_bug = profile_location_longchau_bug.merge(temp_latest_location_longchau, how='left', 
                                                                        on=['cardcode_longchau', 'city'])
    profile_location_longchau = profile_location_longchau[~profile_location_longchau['cardcode_longchau'].isin(profile_location_longchau_bug['cardcode_longchau'])]
    profile_location_longchau = profile_location_longchau.append(profile_location_longchau_bug, 
                                                                 ignore_index=True)

    # normlize address
    profile_longchau['address'] = profile_longchau['address'].str.strip().replace(dict_trash)
    profile_longchau = profile_longchau.drop(columns=['city'])
    profile_longchau = profile_longchau.merge(profile_location_longchau, how='left', on=['cardcode_longchau', 'address'])

    profile_longchau.loc[profile_longchau['street'].isna(), 'street'] = None
    profile_longchau.loc[profile_longchau['ward'].isna(), 'ward'] = None
    profile_longchau.loc[profile_longchau['district'].isna(), 'district'] = None
    profile_longchau.loc[profile_longchau['city'].isna(), 'city'] = None

    ## full address
    columns = ['street', 'ward', 'district', 'city']
    profile_longchau['address'] = profile_longchau[columns].fillna('').agg(', '.join, axis=1).str.replace('(?<![a-zA-Z0-9]),', '', regex=True).str.replace('-(?![a-zA-Z0-9])', '', regex=True)
    profile_longchau['address'] = profile_longchau['address'].str.strip(', ').str.strip(',').str.strip()
    profile_longchau['address'] = profile_longchau['address'].str.strip().replace(dict_trash)
    profile_longchau.loc[profile_longchau['address'].notna(), 'source_address'] = profile_longchau['source_city']

    ## unit_address
    profile_longchau = profile_longchau.rename(columns={'street': 'unit_address'})
    profile_longchau.loc[profile_longchau['unit_address'].notna(), 'source_unit_address'] = 'FSHOP from profile'

    # add info
    profile_longchau['birthday'] = None
    columns = ['cardcode_longchau', 'phone_raw', 'phone', 'is_phone_valid', 
               'email_raw', 'email', 'is_email_valid',
               'name', 'pronoun', 'is_full_name', 'gender', 
               'birthday', 'customer_type', # 'customer_type_detail',
               'address', 'source_address', 'unit_address', 'source_unit_address', 
               'ward', 'source_ward', 'district', 'source_district', 'city', 'source_city']
    profile_longchau = profile_longchau[columns]

    # Fill 'Ca nhan'
    profile_longchau.loc[profile_longchau['name'].notna() & profile_longchau['customer_type'].isna(), 'customer_type'] = 'Ca nhan'

    # Save
    profile_longchau['d'] = date_str
    profile_longchau.drop_duplicates().to_parquet(f'{unify_path}/longchau.parquet', 
                                filesystem=hdfs,
                                index=False,
                                partition_cols='d')
    
if __name__ == '__main__':
    
    date_str = sys.argv[1]
    UnifyLongChau(date_str)
