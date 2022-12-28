
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

def UnifyFo(date_str):
    # VARIABLES
    raw_path = '/data/fpt/ftel/cads/dep_solution/sa/dev/raw'
    unify_path = '/data/fpt/ftel/cads/dep_solution/sa/dev/pre'
    f_group = 'fo'

    dict_trash = {'': None, 'Nan': None, 'nan': None, 'None': None, 'none': None, 'Null': None, 'null': None, "''": None}

    # phone, email (valid)
    valid_phone = pd.read_parquet('/data/fpt/ftel/cads/dep_solution/sa/dev/utils/valid_phone_latest.parquet',
                                  filesystem=hdfs, columns=['phone_raw', 'phone', 'is_phone_valid'])
    valid_email = pd.read_parquet('/data/fpt/ftel/cads/dep_solution/sa/dev/utils/valid_email_latest.parquet',
                                  filesystem=hdfs, columns=['email_raw', 'email', 'is_email_valid'])

    # load profile fo
    profile_fo = pd.read_parquet(f'{raw_path}/{f_group}.parquet/d={date_str}', filesystem=hdfs)

    # info
    profile_fo = profile_fo[['vne_id_fo', 'phone', 'email', 'name', 'gender', 'age', 'address', 'last_active', 'active_date']].copy()
    profile_fo = profile_fo.rename(columns={'vne_id_fo': 'vne_id'})
    profile_fo = profile_fo.sort_values(by=['vne_id', 'last_active', 'active_date'], ascending=False)
    profile_fo = profile_fo.drop_duplicates(subset=['vne_id'], keep='first')
    profile_fo = profile_fo.drop(columns=['last_active', 'active_date'])
    
    # merge get phone, email (valid)
    profile_fo = profile_fo.rename(columns={'phone': 'phone_raw', 'email': 'email_raw'})
    profile_fo = profile_fo.merge(valid_phone, how='left', on=['phone_raw'])
    profile_fo = profile_fo.merge(valid_email, how='left', on=['email_raw'])

    # birthday
    now_year = datetime.today().year
    profile_fo.loc[profile_fo['age'] < 0, 'age'] = np.nan
    profile_fo.loc[profile_fo['age'] > profile_fo['age'].quantile(0.99), 'age'] = np.nan
    profile_fo.loc[profile_fo['age'].notna(), 'birthday'] = '0/0/' + (now_year - profile_fo['age']).astype(str).str.replace('.0', '', regex=False)
    profile_fo = profile_fo.drop(columns=['age'])
    profile_fo.loc[profile_fo['birthday'].isna(), 'birthday'] = None

    # gender
    profile_fo['gender'] = profile_fo['gender'].replace({'Female': 'F', 'Male': 'M', 'Other': None})

    # customer_type
    condition_name = profile_fo['name'].notna()
    profile_fo.loc[condition_name, 'name'] = profile_fo.loc[condition_name, 'name'].apply(clean_name_cdp)
    
    name_fo = profile_fo[profile_fo['name'].notna()][['name']].copy().drop_duplicates()
    name_fo = preprocess_lib.ExtractCustomerType(name_fo)
    profile_fo = profile_fo.merge(name_fo, how='left', on=['name'])
    profile_fo.loc[profile_fo['customer_type'].isna(), 'customer_type'] = None
#     profile_fo.loc[profile_fo['customer_type_detail'].isna(), 'customer_type_detail'] = None

    # drop name is username_email
    profile_fo['username_email'] = profile_fo['email'].str.split('@').str[0]
    profile_fo.loc[profile_fo['name'] == profile_fo['username_email'], 'name'] = None
    profile_fo = profile_fo.drop(columns=['username_email'])

    # clean name
    condition_name = profile_fo['customer_type'].isna() & profile_fo['name'].notna()
    profile_fo.loc[
        condition_name,
        ['clean_name', 'pronoun']
    ] = profile_fo.loc[condition_name, ].apply(lambda row: preprocess_lib.CleanName(row['name'], row['email']), axis=1).tolist()
    profile_fo.loc[profile_fo['customer_type'].isna(), 'name'] = profile_fo['clean_name']
    profile_fo = profile_fo.drop(columns=['clean_name'])
    
    # skip pronoun
    profile_fo['name'] = profile_fo['name'].str.strip().str.title()
    skip_names = ['Vợ', 'Vo', 'Anh', 'Chị', 'Chi', 'Mẹ', 'Me', 'Em', 'Ba', 'Chú', 'Chu', 'Bác', 'Bac', 'Ông', 'Ong', 'Cô', 'Co', 'Cha', 'Dì', 'Dượng']
    profile_fo.loc[profile_fo['name'].isin(skip_names), 'name'] = None

    # format name
    with mp.Pool(8) as pool:
        profile_fo[['last_name', 'middle_name', 'first_name']] = pool.map(preprocess_lib.SplitName, profile_fo['name'])
    columns = ['last_name', 'middle_name', 'first_name']
    profile_fo.loc[condition_name, 'name'] = profile_fo[columns].fillna('').agg(' '.join, axis=1).str.replace('(?<![a-zA-Z0-9]),', '', regex=True).str.replace('-(?![a-zA-Z0-9])', '', regex=True)
    profile_fo['name'] = profile_fo['name'].str.strip().replace(dict_trash)

    # is full name
    profile_fo.loc[profile_fo['last_name'].notna() & profile_fo['first_name'].notna(), 'is_full_name'] = True
    profile_fo['is_full_name'] = profile_fo['is_full_name'].fillna(False)
    profile_fo = profile_fo.drop(columns=['last_name', 'middle_name', 'first_name'])
    profile_fo['name'] = profile_fo['name'].str.strip().str.title()

    # valid gender by model
    profile_fo.loc[profile_fo['customer_type'].notna(), 'gender'] = None
    # profile_fo.loc[profile_fo['gender'].notna() & profile_fo['name'].isna(), 'gender'] = None
    profile_fo_1 = profile_fo[profile_fo['name'].notna()].copy()
    profile_fo_2 = profile_fo[profile_fo['name'].isna()].copy()
    profile_fo_1 = preprocess_lib.Name2Gender(profile_fo_1, name_col='name')
    profile_fo_1.loc[profile_fo_1['gender'].notna() & 
                     (profile_fo_1['gender'] != profile_fo_1['predict_gender']), 'gender'] = None
    profile_fo_1 = profile_fo_1.drop(columns=['predict_gender'])
    profile_fo = pd.concat([profile_fo_1, profile_fo_2], ignore_index=True).sample(frac=1)

    # address, city
    norm_fo_city = pd.read_parquet('/data/fpt/ftel/cads/dep_solution/user/namdp11/scross_fill/runner/refactor/material/ftel_provinces.parquet', 
                                   filesystem=hdfs)
    norm_fo_city.columns = ['city', 'norm_city']
    profile_fo.loc[profile_fo['address'] == 'Not set', 'address'] = None
    profile_fo.loc[profile_fo['address'].notna(), 'city'] = profile_fo.loc[profile_fo['address'].notna(), 'address'].apply(unidecode)
    profile_fo['city'] = profile_fo['city'].replace({'Ba Ria - Vung Tau': 'Vung Tau', 'Thua Thien Hue': 'Hue',
                                                     'Bac Kan': 'Bac Can', 'Dak Nong': 'Dac Nong'})
    profile_fo = profile_fo.merge(norm_fo_city, how='left', on='city')
    profile_fo['city'] = profile_fo['norm_city']
    profile_fo = profile_fo.drop(columns=['norm_city'])
    profile_fo.loc[profile_fo['city'].isna(), 'city'] = None
    profile_fo['address'] = None

    # add info
    profile_fo['unit_address'] = None
    profile_fo['ward'] = None
    profile_fo['district'] = None
    columns = ['vne_id', 'phone_raw', 'phone', 'is_phone_valid', 
               'email_raw', 'email', 'is_email_valid',
               'name', 'pronoun', 'is_full_name', 'gender', 
               'birthday', 'customer_type',  # 'customer_type_detail',
               'address', 'unit_address', 'ward', 'district', 'city']
    profile_fo = profile_fo[columns]
    profile_fo = profile_fo.rename(columns = {'vne_id': 'vne_id_fo'})
    
    # Fill 'Ca nhan'
    profile_fo.loc[profile_fo['name'].notna() & profile_fo['customer_type'].isna(), 'customer_type'] = 'Ca nhan'

    # Save
    profile_fo['d'] = date_str
    profile_fo.drop_duplicates().to_parquet(f'{unify_path}/fo.parquet', 
                          filesystem=hdfs,
                          index=False,
                          partition_cols='d')

    # MOST LOCATION IP
    dict_ip_path = '/data/fpt/ftel/cads/dep_solution/user/namdp11/scross_fill/runner/ip/dictionary'
    log_ip_path = '/data/fpt/ftel/cads/dep_solution/user/namdp11/scross_fill/runner/ip/fo'

    ip_location1 = pd.read_parquet(f'{dict_ip_path}/ip_location_batch_1.parquet', filesystem=hdfs)
    ip_location2 = pd.read_parquet(f'{dict_ip_path}/ip_location_batch_2.parquet', filesystem=hdfs)
    ip_location = ip_location1.append(ip_location2, ignore_index=True)
    ip_location = ip_location[['ip', 'name_province', 'name_district']].copy()

    # update ip
    def IpFo(date):
        date_str = date.strftime('%Y-%m-%d')
        try:
            # load log ip
            log_df = pd.read_parquet(f"/data/fpt/fdp/fo/dwh/stag_access_features.parquet/d={date_str}", 
                                     filesystem=hdfs, columns=['user_id', 'ip', 'isp']).drop_duplicates()
            log_df['date'] = date_str
            log_df.to_parquet(f'{log_ip_path}/ip_{date_str}.parquet', index=False, filesystem=hdfs)

            # add location
            location_df = log_df.merge(ip_location, how='left', on='ip')
            location_df.to_parquet(f'{log_ip_path}/location/ip_{date_str}.parquet', index=False, filesystem=hdfs)
        except:
            print('IP-FO Fail: {}'.format(date_str))

    start_date = sorted([f.path
                         for f in hdfs.get_file_info(fs.FileSelector(log_ip_path))
                        ])[-2][-18:-8]
    end_date = (datetime.today() - timedelta(days=1)).strftime('%Y-%m-%d')
    dates = pd.date_range(start_date, end_date, freq='D')

    for date in dates:
        IpFo(date)

    # stats location ip
    logs_ip_path = sorted([f.path for f in hdfs.get_file_info(fs.FileSelector(f'{log_ip_path}/location/'))])[-180:]
    ip_fo = pd.read_parquet(logs_ip_path, filesystem=hdfs)
    stats_ip_fo = ip_fo.groupby(by=['user_id', 'name_province', 'name_district'])['date'].agg(num_date='count').reset_index()
    stats_ip_fo = stats_ip_fo.sort_values(by=['user_id', 'num_date'], ascending=False)
    most_ip_fo = stats_ip_fo.drop_duplicates(subset=['user_id'], keep='first')
    most_ip_fo.to_parquet('/data/fpt/ftel/cads/dep_solution/sa/dev/utils/fo_location_most.parquet', 
                          index=False, filesystem=hdfs)

if __name__ == '__main__':
    
    date_str = sys.argv[1]
    UnifyFo(date_str)
