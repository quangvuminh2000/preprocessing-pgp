
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

def UnifyFsoft(date_str):
    # VARIABLES
    raw_path = ROOT_PATH + '/raw'
    unify_path = ROOT_PATH + '/pre'
    f_group = 'fsoft'

    dict_trash = {'': None, 'Nan': None, 'nan': None, 'None': None, 'none': None, 'Null': None, 'null': None, "''": None}

    # phone, email(valid)
    valid_phone = pd.read_parquet(ROOT_PATH + '/utils/valid_phone_latest.parquet',
                                  filesystem=hdfs, columns=['phone_raw', 'phone', 'is_phone_valid'])
    valid_email = pd.read_parquet(ROOT_PATH + '/utils/valid_email_latest.parquet',
                                  filesystem=hdfs, columns=['email_raw', 'email', 'is_email_valid'])

    # load profile fsoft
    profile_fsoft = pd.read_parquet(f'{raw_path}/{f_group}.parquet/d={date_str}', filesystem=hdfs)

    # info
    profile_fsoft = profile_fsoft[['fsoft_id', 'phone', 'email', 'name', 'birthday', 'address', 'city', 'district', 'customer_type']].copy()
    profile_fsoft = profile_fsoft.rename(columns={'customer_type': 'customer_type_fsoft', 'phone': 'phone_raw', 'email': 'email_raw'})
    # profile_fsoft.loc[profile_fsoft['gender'] == '-1', 'gender'] = None
    profile_fsoft.loc[profile_fsoft['address'].isin(['', 'Null', 'None', 'Test']), 'address'] = None
    profile_fsoft.loc[profile_fsoft['address'].notna() & profile_fsoft['address'].str.isnumeric(), 'address'] = None
    profile_fsoft.loc[profile_fsoft['address'].str.len() < 5, 'address'] = None
    profile_fsoft['customer_type_fsoft'] = profile_fsoft['customer_type_fsoft'].replace({'Individual': 'Ca nhan', 
                                                                                         'Company': 'Cong ty', 
                                                                                         'Other': None})
    # merge get phone(valid)
    profile_fsoft = profile_fsoft.merge(valid_phone, how='left', on=['phone_raw'])
    profile_fsoft = profile_fsoft.merge(valid_email, how='left', on=['email_raw'])
    
    # customer_type
    condition_name = profile_fsoft['name'].notna()
    profile_fsoft.loc[condition_name, 'name'] = profile_fsoft.loc[condition_name, 'name'].apply(clean_name_cdp)
    
    name_fsoft = profile_fsoft[profile_fsoft['name'].notna()][['name']].copy().drop_duplicates()
    name_fsoft = preprocess_lib.ExtractCustomerType(name_fsoft)
    profile_fsoft = profile_fsoft.merge(name_fsoft, how='left', on=['name'])
    profile_fsoft.loc[profile_fsoft['customer_type'].isna(), 'customer_type'] = profile_fsoft['customer_type_fsoft']
    profile_fsoft = profile_fsoft.drop(columns=['customer_type_fsoft'])
    profile_fsoft.loc[profile_fsoft['customer_type'].isna(), 'customer_type'] = None

    # drop name is username_email
    profile_fsoft['username_email'] = profile_fsoft['email'].str.split('@').str[0]
    profile_fsoft.loc[profile_fsoft['name'] == profile_fsoft['username_email'], 'name'] = None
    profile_fsoft = profile_fsoft.drop(columns=['username_email'])
    
    # clean name
    condition_name = profile_fsoft['customer_type'].isin([None, 'Ca nhan', np.nan]) & profile_fsoft['name'].notna()
    profile_fsoft.loc[
        condition_name,
        ['clean_name', 'pronoun']
    ] = profile_fsoft.loc[condition_name, ].apply(lambda row: preprocess_lib.CleanName(row['name'], row['email']), axis=1).tolist()
    profile_fsoft.loc[profile_fsoft['customer_type'].isin([None, 'Ca nhan', np.nan]), 'name'] = profile_fsoft['clean_name']
    profile_fsoft = profile_fsoft.drop(columns=['clean_name'])
    
    # skip pronoun
    profile_fsoft['name'] = profile_fsoft['name'].str.strip().str.title()
    skip_names = ['Vợ', 'Vo', 'Anh', 'Chị', 'Chi', 'Mẹ', 'Me', 'Em', 'Ba', 'Chú', 'Chu', 'Bác', 'Bac', 'Ông', 'Ong', 'Cô', 'Co', 'Cha', 'Dì', 'Dượng']
    profile_fsoft.loc[profile_fsoft['name'].isin(skip_names), 'name'] = None

    # format name
    with mp.Pool(8) as pool:
        profile_fsoft[['last_name', 'middle_name', 'first_name']] = pool.map(preprocess_lib.SplitName, profile_fsoft['name'])
    columns = ['last_name', 'middle_name', 'first_name']
    profile_fsoft.loc[condition_name, 'name'] = profile_fsoft[columns].fillna('').agg(' '.join, axis=1).str.replace('(?<![a-zA-Z0-9]),', '', regex=True).str.replace('-(?![a-zA-Z0-9])', '', regex=True)
    profile_fsoft['name'] = profile_fsoft['name'].str.strip().replace(dict_trash)
    profile_fsoft.loc[profile_fsoft['name'].isna(), 'name'] = None

    # is full name
    profile_fsoft.loc[profile_fsoft['last_name'].notna() & profile_fsoft['first_name'].notna(), 'is_full_name'] = True
    profile_fsoft['is_full_name'] = profile_fsoft['is_full_name'].fillna(False)
    profile_fsoft = profile_fsoft.drop(columns=['last_name', 'middle_name', 'first_name'])
    profile_fsoft['name'] = profile_fsoft['name'].str.strip().str.title()

    # valid gender by model
    # profile_fsoft.loc[profile_fsoft['customer_type'] != 'Ca nhan', 'gender'] = None
    profile_fsoft_1 = profile_fsoft[profile_fsoft['name'].notna()].copy()
    profile_fsoft_2 = profile_fsoft[profile_fsoft['name'].isna()].copy()
    profile_fsoft_1 = preprocess_lib.Name2Gender(profile_fsoft_1, name_col='name')
    # profile_fsoft_1.loc[profile_fsoft_1['gender'].notna() & 
    #                     (profile_fsoft_1['gender'] != profile_fsoft_1['predict_gender']), 'gender'] = None
    # profile_fsoft_1 = profile_fsoft_1.drop(columns=['predict_gender'])
    profile_fsoft_1.rename(columns={'predict_gender': 'gender'}, inplace=True)
    profile_fsoft_2['gender'] = None
    profile_fsoft = pd.concat([profile_fsoft_1, profile_fsoft_2], ignore_index=True).sample(frac=1)
    
    # normlize address
    profile_fsoft['address'] = profile_fsoft['address'].str.strip().replace(dict_trash)
    profile_fsoft['street'] = None
    profile_fsoft['ward'] = None

    # ## full address
    # columns = ['street', 'ward', 'district', 'city']
    # profile_fsoft['address'] = profile_fsoft[columns].fillna('').agg(', '.join, axis=1).str.replace('(?<![a-zA-Z0-9]),', '', regex=True).str.replace('-(?![a-zA-Z0-9])', '', regex=True)
    # profile_fsoft['address'] = profile_fsoft['address'].str.strip(', ').str.strip(',').str.strip()
    # profile_fsoft['address'] = profile_fsoft['address'].str.strip().replace(dict_trash)
    # profile_fsoft.loc[profile_fsoft['address'].notna(), 'source_address'] = profile_fsoft['source_city']

    ## unit_address
    profile_fsoft = profile_fsoft.rename(columns={'street': 'unit_address'})
    profile_fsoft.loc[profile_fsoft['unit_address'].notna(), 'source_unit_address'] = 'FSOFT from profile'
    profile_fsoft.loc[profile_fsoft['ward'].notna(), 'source_ward'] = 'FSOFT from profile'
    profile_fsoft.loc[profile_fsoft['district'].notna(), 'source_district'] = 'FSOFT from profile'
    profile_fsoft.loc[profile_fsoft['city'].notna(), 'source_city'] = 'FSOFT from profile'
    profile_fsoft.loc[profile_fsoft['address'].notna(), 'source_address'] = 'FSOFT from profile'

    # add info
    columns = ['fsoft_id', 'phone_raw', 'phone', 'is_phone_valid', 
               'email_raw', 'email', 'is_email_valid',
               'name', 'pronoun', 'is_full_name', 'gender', 
               'birthday', 'customer_type', # 'customer_type_detail',
               'address', 'source_address', 'unit_address', 'source_unit_address', 
               'ward', 'source_ward', 'district', 'source_district', 'city', 'source_city']
    profile_fsoft = profile_fsoft[columns]

    # Fill 'Ca nhan'
    profile_fsoft.loc[profile_fsoft['name'].notna() & profile_fsoft['customer_type'].isna(), 'customer_type'] = 'Ca nhan'

    # Save
    profile_fsoft['d'] = date_str
    profile_fsoft.drop_duplicates().to_parquet(f'{unify_path}/fsoft.parquet', 
                             filesystem=hdfs,
                             index=False,
                             partition_cols='d')
    
if __name__ == '__main__':
    
    date_str = sys.argv[1]
    UnifyFsoft(date_str)
