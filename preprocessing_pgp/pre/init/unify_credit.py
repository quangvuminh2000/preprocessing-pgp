
import pandas as pd
from glob import glob
import numpy as np
from unidecode import unidecode
from string import punctuation
from datetime import datetime, timedelta
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
ROOT_PATH = '/data/fpt/ftel/cads/dep_solution/sa/cdp/core'

def UnifyCredit(date_str):
    # VARIABLES
    raw_path = ROOT_PATH + '/raw'
    unify_path = ROOT_PATH + '/pre'
    f_group = 'credit'

    dict_trash = {'': None, 'Nan': None, 'nan': None, 'None': None, 'none': None, 'Null': None, 'null': None, "''": None}

    # phone(valid)
    valid_phone = pd.read_parquet(ROOT_PATH + '/utils/valid_phone_latest.parquet',
                                  filesystem=hdfs, columns=['phone_raw', 'phone', 'is_phone_valid'])

    # load profile credit
    profile_credit = pd.read_parquet(f'{raw_path}/{f_group}.parquet/d={date_str}', filesystem=hdfs)

    # info
    profile_credit = profile_credit[['cardcode_fshop', 'phone', 'name', 'gender', 'birthday', 'address', 'customer_type']].copy()
    
    profile_credit['email'] = None
    
    profile_credit = profile_credit.rename(columns={'customer_type': 'customer_type_credit', 'phone': 'phone_raw'})
    profile_credit.loc[profile_credit['gender'] == '-1', 'gender'] = None
    profile_credit.loc[profile_credit['address'].isin(['', 'Null', 'None', 'Test']), 'address'] = None
    profile_credit.loc[profile_credit['address'].notna() & profile_credit['address'].str.isnumeric(), 'address'] = None
    profile_credit.loc[profile_credit['address'].str.len() < 5, 'address'] = None
    profile_credit['customer_type_credit'] = profile_credit['customer_type_credit'].replace({'Individual': 'Ca nhan', 
                                                                                         'Company': 'Cong ty', 
                                                                                         'Other': None})
    # merge get phone(valid)
    profile_credit = profile_credit.merge(valid_phone, how='left', on=['phone_raw'])
    
    # customer_type
    condition_name = profile_credit['name'].notna()
    profile_credit.loc[condition_name, 'name'] = profile_credit.loc[condition_name, 'name'].apply(clean_name_cdp)
    
    name_credit = profile_credit[profile_credit['name'].notna()][['name']].copy().drop_duplicates()
    name_credit = preprocess_lib.ExtractCustomerType(name_credit)
    profile_credit = profile_credit.merge(name_credit, how='left', on=['name'])
    profile_credit.loc[profile_credit['customer_type'].isna(), 'customer_type'] = profile_credit['customer_type_credit']
    profile_credit = profile_credit.drop(columns=['customer_type_credit'])
    profile_credit.loc[profile_credit['customer_type'].isna(), 'customer_type'] = None

    # clean name
    condition_name = profile_credit['customer_type'].isin([None, 'Ca nhan', np.nan]) & profile_credit['name'].notna()
    profile_credit.loc[
        condition_name,
        ['clean_name', 'pronoun']
    ] = profile_credit.loc[condition_name, ].apply(lambda row: preprocess_lib.CleanName(row['name'], row['email']), axis=1).tolist()
    profile_credit.loc[profile_credit['customer_type'].isin([None, 'Ca nhan', np.nan]), 'name'] = profile_credit['clean_name']
    profile_credit = profile_credit.drop(columns=['clean_name'])
    
    # skip pronoun
    profile_credit['name'] = profile_credit['name'].str.strip().str.title()
    skip_names = ['Vợ', 'Vo', 'Anh', 'Chị', 'Chi', 'Mẹ', 'Me', 'Em', 'Ba', 'Chú', 'Chu', 'Bác', 'Bac', 'Ông', 'Ong', 'Cô', 'Co', 'Cha', 'Dì', 'Dượng']
    profile_credit.loc[profile_credit['name'].isin(skip_names), 'name'] = None

    # format name
    with mp.Pool(8) as pool:
        profile_credit[['last_name', 'middle_name', 'first_name']] = pool.map(preprocess_lib.SplitName, profile_credit['name'])
    columns = ['last_name', 'middle_name', 'first_name']
    profile_credit.loc[condition_name, 'name'] = profile_credit[columns].fillna('').agg(' '.join, axis=1).str.replace('(?<![a-zA-Z0-9]),', '', regex=True).str.replace('-(?![a-zA-Z0-9])', '', regex=True)
    profile_credit['name'] = profile_credit['name'].str.strip().replace(dict_trash)
    profile_credit.loc[profile_credit['name'].isna(), 'name'] = None

    # is full name
    profile_credit.loc[profile_credit['last_name'].notna() & profile_credit['first_name'].notna(), 'is_full_name'] = True
    profile_credit['is_full_name'] = profile_credit['is_full_name'].fillna(False)
    profile_credit = profile_credit.drop(columns=['last_name', 'middle_name', 'first_name'])
    profile_credit['name'] = profile_credit['name'].str.strip().str.title()

    # valid gender by model
    profile_credit.loc[profile_credit['customer_type'] != 'Ca nhan', 'gender'] = None
    profile_credit_1 = profile_credit[profile_credit['name'].notna()].copy()
    profile_credit_2 = profile_credit[profile_credit['name'].isna()].copy()
    profile_credit_1 = preprocess_lib.Name2Gender(profile_credit_1, name_col='name')
    profile_credit_1.loc[profile_credit_1['gender'].notna() & 
                        (profile_credit_1['gender'] != profile_credit_1['predict_gender']), 'gender'] = None
    profile_credit_1 = profile_credit_1.drop(columns=['predict_gender'])
    profile_credit_2['gender'] = None
    profile_credit = pd.concat([profile_credit_1, profile_credit_2], ignore_index=True).sample(frac=1)
    
    # normlize address
    profile_credit['address'] = profile_credit['address'].str.strip().replace(dict_trash)
    profile_credit['street'] = None
    profile_credit['ward'] = None
    profile_credit['district'] = None
    profile_credit['city'] = None

    # ## full address
    # columns = ['street', 'ward', 'district', 'city']
    # profile_credit['address'] = profile_credit[columns].fillna('').agg(', '.join, axis=1).str.replace('(?<![a-zA-Z0-9]),', '', regex=True).str.replace('-(?![a-zA-Z0-9])', '', regex=True)
    # profile_credit['address'] = profile_credit['address'].str.strip(', ').str.strip(',').str.strip()
    # profile_credit['address'] = profile_credit['address'].str.strip().replace(dict_trash)
    # profile_credit.loc[profile_credit['address'].notna(), 'source_address'] = profile_credit['source_city']

    ## unit_address
    profile_credit = profile_credit.rename(columns={'street': 'unit_address'})
    profile_credit.loc[profile_credit['unit_address'].notna(), 'source_unit_address'] = 'CREDIT from profile'
    profile_credit.loc[profile_credit['ward'].notna(), 'source_ward'] = 'CREDIT from profile'
    profile_credit.loc[profile_credit['district'].notna(), 'source_district'] = 'CREDIT from profile'
    profile_credit.loc[profile_credit['city'].notna(), 'source_city'] = 'CREDIT from profile'
    profile_credit.loc[profile_credit['address'].notna(), 'source_address'] = 'CREDIT from profile'

    # add info
    columns = ['cardcode_fshop', 'phone_raw', 'phone', 'is_phone_valid', 
               # 'email_raw', 'email', 'is_email_valid',
               'name', 'pronoun', 'is_full_name', 'gender', 
               'birthday', 'customer_type', # 'customer_type_detail',
               'address', 'source_address', 'unit_address', 'source_unit_address', 
               'ward', 'source_ward', 'district', 'source_district', 'city', 'source_city']
    profile_credit = profile_credit[columns]

    # Fill 'Ca nhan'
    profile_credit.loc[profile_credit['name'].notna() & profile_credit['customer_type'].isna(), 'customer_type'] = 'Ca nhan'

    # Save
    profile_credit['d'] = date_str
    profile_credit.drop_duplicates().to_parquet(f'{unify_path}/credit.parquet', 
                             filesystem=hdfs,
                             index=False,
                             partition_cols='d')
    
if __name__ == '__main__':
    
    date_str = sys.argv[1]
    UnifyCredit(date_str)
