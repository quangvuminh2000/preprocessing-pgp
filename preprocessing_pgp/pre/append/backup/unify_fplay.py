
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

# function get profile change/new
def DifferenceProfile(now_df, yesterday_df):
    difference_df = now_df[~now_df.apply(tuple,1).isin(yesterday_df.apply(tuple,1))].copy()
    return difference_df

# function unify profile
def UnifyFplay(profile_fplay, date_str='2022-07-01'):
    # VARIABLE
    dict_trash = {'': None, 'Nan': None, 'nan': None, 'None': None, 'none': None, 'Null': None, 'null': None, "''": None}

    # phone, email (valid)
    valid_phone = pd.read_parquet('/data/fpt/ftel/cads/dep_solution/sa/dev/utils/valid_phone_latest.parquet',
                                  filesystem=hdfs, columns=['phone_raw', 'phone', 'is_phone_valid'])
    valid_email = pd.read_parquet('/data/fpt/ftel/cads/dep_solution/sa/dev/utils/valid_email_latest.parquet',
                                  filesystem=hdfs, columns=['email_raw', 'email', 'is_email_valid'])

    # info
    profile_fplay = profile_fplay.rename(columns={'user_id_fplay': 'user_id'})
    profile_fplay = profile_fplay.sort_values(by=['user_id', 'last_active', 'active_date'], ascending=False)
    profile_fplay = profile_fplay.drop_duplicates(subset=['user_id'], keep='first')
    profile_fplay = profile_fplay.drop(columns=['last_active', 'active_date'])
    
    # merge get phone, email (valid)
    profile_fplay = profile_fplay.rename(columns={'phone': 'phone_raw', 'email': 'email_raw'})
    profile_fplay = profile_fplay.merge(valid_phone, how='left', on=['phone_raw'])
    profile_fplay = profile_fplay.merge(valid_email, how='left', on=['email_raw'])

    # customer_type
    condition_name = profile_fplay['name'].notna()
    profile_fplay.loc[condition_name, 'name'] = profile_fplay.loc[condition_name, 'name'].apply(clean_name_cdp)
    
    name_fplay = profile_fplay[profile_fplay['name'].notna()][['name']].copy().drop_duplicates()
    name_fplay = preprocess_lib.ExtractCustomerType(name_fplay)
    profile_fplay = profile_fplay.merge(name_fplay, how='left', on=['name'])
    profile_fplay.loc[profile_fplay['customer_type'].isna(), 'customer_type'] = None
#     profile_fplay.loc[profile_fplay['customer_type_detail'].isna(), 'customer_type_detail'] = None

    # drop name is username_email
    profile_fplay['username_email'] = profile_fplay['email'].str.split('@').str[0]
    profile_fplay.loc[profile_fplay['name'] == profile_fplay['username_email'], 'name'] = None
    profile_fplay = profile_fplay.drop(columns=['username_email'])

    # clean name
    condition_name = profile_fplay['customer_type'].isna() & profile_fplay['name'].notna()
    profile_fplay.loc[
        condition_name,
        ['clean_name', 'pronoun']
    ] = profile_fplay.loc[condition_name, ].apply(lambda row: preprocess_lib.CleanName(row['name'], row['email']), axis=1).tolist()
    profile_fplay.loc[profile_fplay['customer_type'].isna(), 'name'] = profile_fplay['clean_name']
    profile_fplay = profile_fplay.drop(columns=['clean_name'])

    # skip pronoun
    profile_fplay['name'] = profile_fplay['name'].str.strip().str.title()
    skip_names = ['Vợ', 'Vo', 'Anh', 'Chị', 'Chi', 'Mẹ', 'Me', 'Em', 'Ba', 'Chú', 'Chu', 'Bác', 'Bac', 'Ông', 'Ong', 'Cô', 'Co', 'Cha', 'Dì', 'Dượng']
    profile_fplay.loc[profile_fplay['name'].isin(skip_names), 'name'] = None
    
    # format name
    with mp.Pool(8) as pool:
        profile_fplay[['last_name', 'middle_name', 'first_name']] = pool.map(preprocess_lib.SplitName, profile_fplay['name'])
    columns = ['last_name', 'middle_name', 'first_name']
    profile_fplay.loc[condition_name, 'name'] = profile_fplay[columns].fillna('').agg(' '.join, axis=1).str.replace('(?<![a-zA-Z0-9]),', '', regex=True).str.replace('-(?![a-zA-Z0-9])', '', regex=True)
    profile_fplay['name'] = profile_fplay['name'].str.strip().replace(dict_trash)

    # is full name
    profile_fplay.loc[profile_fplay['last_name'].notna() & profile_fplay['first_name'].notna(), 'is_full_name'] = True
    profile_fplay['is_full_name'] = profile_fplay['is_full_name'].fillna(False)
    profile_fplay = profile_fplay.drop(columns=['last_name', 'middle_name', 'first_name'])
    profile_fplay['name'] = profile_fplay['name'].str.strip().str.title()

    # add info
    profile_fplay['gender'] = None
    profile_fplay['birthday'] = None
    profile_fplay['address'] = None
    profile_fplay['unit_address'] = None
    profile_fplay['ward'] = None
    profile_fplay['district'] = None
    profile_fplay['city'] = None
    columns = ['user_id', 'phone_raw', 'phone', 'is_phone_valid', 
               'email_raw', 'email', 'is_email_valid',
               'name', 'pronoun', 'is_full_name', 'gender', 
               'birthday', 'customer_type', # 'customer_type_detail',
               'address', 'unit_address', 'ward', 'district', 'city']
    profile_fplay = profile_fplay[columns]
    profile_fplay = profile_fplay.rename(columns = {'user_id': 'user_id_fplay'})

    # Fill 'Ca nhan'
    profile_fplay.loc[profile_fplay['name'].notna() & profile_fplay['customer_type'].isna(), 'customer_type'] = 'Ca nhan'
    
    # return
    return profile_fplay
    
# function update profile (unify)
def UpdateUnifyFplay(now_str):
    # VARIABLES
    raw_path = '/data/fpt/ftel/cads/dep_solution/sa/dev/raw'
    unify_path = '/data/fpt/ftel/cads/dep_solution/sa/dev/pre'
    f_group = 'fplay'
    yesterday_str = (datetime.strptime(now_str, '%Y-%m-%d') - timedelta(days=1)).strftime('%Y-%m-%d')

    # load profile (yesterday, now)
    info_columns = ['user_id_fplay', 'phone', 'email', 'name', 'last_active', 'active_date']
    now_profile = pd.read_parquet(f'{raw_path}/{f_group}.parquet/d={now_str}', 
                                  filesystem=hdfs, columns=info_columns).drop_duplicates()
    yesterday_profile = pd.read_parquet(f'{raw_path}/{f_group}.parquet/d={yesterday_str}', 
                                        filesystem=hdfs, columns=info_columns).drop_duplicates()
    
    # get profile change/new
    difference_profile = DifferenceProfile(now_profile, yesterday_profile)
    
    # update profile
    profile_unify = pd.DataFrame()
    if difference_profile.empty:
        profile_unify = pd.read_parquet(f'{unify_path}/{f_group}.parquet/d={yesterday_str}', filesystem=hdfs).drop_duplicates()
    else:
        # get profile unify (old + new)
        old_profile_unify = pd.read_parquet(f'{unify_path}/{f_group}.parquet/d={yesterday_str}', filesystem=hdfs)
        new_profile_unify = UnifyFplay(difference_profile, date_str=yesterday_str)

        # synthetic profile
        profile_unify = new_profile_unify.append(old_profile_unify, ignore_index=True)

    # update valid: phone & email
    profile_unify = profile_unify.drop(columns=['phone', 'email', 'is_phone_valid', 'is_email_valid'])

    valid_phone = pd.read_parquet('/data/fpt/ftel/cads/dep_solution/sa/dev/utils/valid_phone_latest.parquet',
                                  filesystem=hdfs, columns=['phone_raw', 'phone', 'is_phone_valid'])
    valid_email = pd.read_parquet('/data/fpt/ftel/cads/dep_solution/sa/dev/utils/valid_email_latest.parquet',
                                  filesystem=hdfs, columns=['email_raw', 'email', 'is_email_valid'])

    profile_unify = profile_unify.merge(valid_phone, how='left', on=['phone_raw'])
    profile_unify = profile_unify.merge(valid_email, how='left', on=['email_raw'])
    profile_unify['is_phone_valid'] = profile_unify['is_phone_valid'].fillna(False)
    profile_unify['is_email_valid'] = profile_unify['is_email_valid'].fillna(False)
        
    # arrange columns
    columns = ['user_id_fplay', 'phone_raw', 'phone', 'is_phone_valid', 
               'email_raw', 'email', 'is_email_valid',
               'name', 'pronoun', 'is_full_name', 'gender', 
               'birthday', 'customer_type', # 'customer_type_detail',
               'address', 'unit_address', 'ward', 'district', 'city']
    
    profile_unify = profile_unify[columns].copy()
    profile_unify = profile_unify.drop_duplicates(subset=['user_id_fplay', 'phone_raw', 'email_raw'], keep='first')
        
    # save
    profile_unify['d'] = now_str
    profile_unify.drop_duplicates().to_parquet(f'{unify_path}/{f_group}.parquet', 
                                               filesystem=hdfs,
                                               index=False,
                                               partition_cols='d')

# function update ip (most)
def UnifyLocationIpFplay():
    # MOST LOCATION IP
    dict_ip_path = '/data/fpt/ftel/cads/dep_solution/user/namdp11/scross_fill/runner/ip/dictionary'
    log_ip_path = '/data/fpt/ftel/cads/dep_solution/user/namdp11/scross_fill/runner/ip/fplay'

    ip_location1 = pd.read_parquet(f'{dict_ip_path}/ip_location_batch_1.parquet', filesystem=hdfs)
    ip_location2 = pd.read_parquet(f'{dict_ip_path}/ip_location_batch_2.parquet', filesystem=hdfs)
    ip_location = ip_location1.append(ip_location2, ignore_index=True)
    ip_location = ip_location[['ip', 'name_province', 'name_district']].copy()

    # update ip
    def IpFplay(date):
        date_str = date.strftime('%Y-%m-%d')
        try:
            # load log ip
            log_df = pd.read_parquet(f'/data/fpt/ftel/fplay/dwh/ds_network.parquet/d={date_str}', 
                                     filesystem=hdfs, columns=['user_id', 'ip', 'isp', 'network_type']).drop_duplicates()
            log_df['date'] = date_str
            log_df.to_parquet(f'{log_ip_path}/ip_{date_str}.parquet', index=False, filesystem=hdfs)

            # add location
            location_df = log_df.merge(ip_location, how='left', on='ip')
            location_df.to_parquet(f'{log_ip_path}/location/ip_{date_str}.parquet', index=False, filesystem=hdfs)

        except:
            print('IP-FPLAY Fail: {}'.format(date_str))


    start_date = sorted([f.path 
                         for f in hdfs.get_file_info(fs.FileSelector(log_ip_path))
                        ])[-2][-18:-8]
    end_date = (datetime.today() - timedelta(days=1)).strftime('%Y-%m-%d')
    dates = pd.date_range(start_date, end_date, freq='D')


    for date in dates:
        IpFplay(date)

    # stats location ip
    logs_ip_path = sorted([f.path for f in hdfs.get_file_info(fs.FileSelector(f'{log_ip_path}/location/'))])[-180:]
    ip_fplay = pd.read_parquet(logs_ip_path, filesystem=hdfs)
    stats_ip_fplay = ip_fplay.groupby(by=['user_id', 'name_province', 'name_district'])['date'].agg(num_date='count').reset_index()
    stats_ip_fplay = stats_ip_fplay.sort_values(by=['user_id', 'num_date'], ascending=False)
    most_ip_fplay = stats_ip_fplay.drop_duplicates(subset=['user_id'], keep='first')
    most_ip_fplay.to_parquet('/data/fpt/ftel/cads/dep_solution/sa/dev/utils/fplay_location_most.parquet', 
                             index=False, filesystem=hdfs)
    
if __name__ == '__main__':
    
    now_str = sys.argv[1]
    UpdateUnifyFplay(now_str)
    UnifyLocationIpFplay()
