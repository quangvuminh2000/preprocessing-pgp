
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

def UnifySendo(date_str):
    # VARIABLES
    raw_path = '/data/fpt/ftel/cads/dep_solution/sa/dev/raw'
    unify_path = '/data/fpt/ftel/cads/dep_solution/sa/dev/pre'
    f_group = 'sendo'

    dict_trash = {'': None, 'Nan': None, 'nan': None, 'None': None, 'none': None, 'Null': None, 'null': None, "''": None}
    dict_location = pd.read_parquet('/data/fpt/ftel/cads/dep_solution/user/namdp11/scross_fill/runner/refactor/material/dict_location.parquet', 
                                    filesystem=hdfs)

    # phone, email (valid)
    valid_phone = pd.read_parquet('/data/fpt/ftel/cads/dep_solution/sa/dev/utils/valid_phone_latest.parquet',
                                  filesystem=hdfs, columns=['phone_raw', 'phone', 'is_phone_valid'])
    valid_email = pd.read_parquet('/data/fpt/ftel/cads/dep_solution/sa/dev/utils/valid_email_latest.parquet',
                                  filesystem=hdfs, columns=['email_raw', 'email', 'is_email_valid'])

    # load profile psendo
    profile_sendo = pd.read_parquet(f'{raw_path}/{f_group}.parquet/d={date_str}', filesystem=hdfs)

    # info
    profile_sendo = profile_sendo[['id_sendo', 'phone', 'email', 'name',
                                   'address', 'ward', 'district', 'city', 'source']].copy()

    # merge get phone, email (valid)
    profile_sendo = profile_sendo.rename(columns={'phone': 'phone_raw', 'email': 'email_raw'})
    profile_sendo = profile_sendo.merge(valid_phone, how='left', on=['phone_raw'])
    profile_sendo = profile_sendo.merge(valid_email, how='left', on=['email_raw'])
    
    # customer_type
    condition_name = profile_sendo['name'].notna()
    profile_sendo.loc[condition_name, 'name'] = profile_sendo.loc[condition_name, 'name'].apply(clean_name_cdp)
    
    name_sendo = profile_sendo[profile_sendo['name'].notna()][['name']].copy().drop_duplicates()
    name_sendo = preprocess_lib.ExtractCustomerType(name_sendo)
    profile_sendo = profile_sendo.merge(name_sendo, how='left', on=['name'])
    profile_sendo.loc[profile_sendo['customer_type'].isna(), 'customer_type'] = None
#     profile_sendo.loc[profile_sendo['customer_type_detail'].isna(), 'customer_type_detail'] = None

    # drop name is username_email
    profile_sendo['username_email'] = profile_sendo['email'].str.split('@').str[0]
    profile_sendo.loc[profile_sendo['name'] == profile_sendo['username_email'], 'name'] = None
    profile_sendo = profile_sendo.drop(columns=['username_email'])

    # clean name
    condition_name = profile_sendo['customer_type'].isna() & profile_sendo['name'].notna()
    profile_sendo.loc[
        condition_name,
        ['clean_name', 'pronoun']
    ] = profile_sendo.loc[condition_name, ].apply(lambda row: preprocess_lib.CleanName(row['name'], row['email']), axis=1).tolist()
    profile_sendo.loc[profile_sendo['customer_type'].isna() , 'name'] = profile_sendo['clean_name']
    profile_sendo = profile_sendo.drop(columns=['clean_name'])

    # skip pronoun
    profile_sendo['name'] = profile_sendo['name'].str.strip().str.title()
    skip_names = ['Vợ', 'Vo', 'Anh', 'Chị', 'Chi', 'Mẹ', 'Me', 'Em', 'Ba', 'Chú', 'Chu', 'Bác', 'Bac', 'Ông', 'Ong', 'Cô', 'Co', 'Cha', 'Dì', 'Dượng']
    profile_sendo.loc[profile_sendo['name'].isin(skip_names), 'name'] = None
    
    # format name
    with mp.Pool(8) as pool:
        profile_sendo[['last_name', 'middle_name', 'first_name']] = pool.map(preprocess_lib.SplitName, profile_sendo['name'])
    columns = ['last_name', 'middle_name', 'first_name']
    profile_sendo.loc[condition_name, 'name'] = profile_sendo[columns].fillna('').agg(' '.join, axis=1).str.replace('(?<![a-zA-Z0-9]),', '', regex=True).str.replace('-(?![a-zA-Z0-9])', '', regex=True)
    profile_sendo['name'] = profile_sendo['name'].str.strip().replace(dict_trash)
    profile_sendo.loc[profile_sendo['name'].isna(), 'name'] = None

    # is full name
    profile_sendo.loc[profile_sendo['last_name'].notna() & profile_sendo['first_name'].notna(), 'is_full_name'] = True
    profile_sendo['is_full_name'] = profile_sendo['is_full_name'].fillna(False)
    profile_sendo = profile_sendo.drop(columns=['last_name', 'middle_name', 'first_name'])
    profile_sendo['name'] = profile_sendo['name'].str.strip().str.title()

    ## spare unit_address
    def SparseUnitAddress(address, ward, district, city):
        result = address.title()

        if ward != None:
            ward = ward.title()

            if 'Xa' in ward:
                for key in ['Xa ', 'Xa', ',', '.', '']:
                    key_ward = ward.replace('Xa ', key).strip()
                    result = result.split(key_ward)[0]
            if 'Phuong' in ward:
                for key in ['Phuong ', 'Phuong', 'P', 'F', 'P.', 'F.', 'F ', 'P ', ',', '.', '']:
                    key_ward = ward.replace('Phuong ', key).strip().replace('0', '')
                    if key_ward.isdigit():
                        continue
                    result = result.split(key_ward)[0]
            elif 'Thi Tran' in ward:
                for key in ['Thi Tran ', 'Thi Tran', 'Tt ', 'Tt.', ',', '.', '']:
                    key_ward = ward.replace('Thi Tran ', key).strip()
                    result = result.split(key_ward)[0]

        # district
        if district != None:
            district = district.title()

            if 'Huyen' in district:
                for key in ['Huyen ', 'Huyen', 'H ', 'H.', ',', '.', '']:
                    key_district = district.replace('Huyen ', key).strip()
                    result = result.split(key_district)[0] 
            elif 'Thi Xa' in district:
                for key in ['Thi Xa ', 'Thi xa', 'Tx ', 'Tx.', ',', '.', '']:
                    key_district = district.replace('Thi Xa ', key).strip()
                    result = result.split(key_district)[0]
            elif 'Quan' in district:
                for key in ['Quan ', 'Quan', 'Q', 'Q.', ',', '.', '']:
                    key_district = district.replace('Quan ', key).strip().replace('0', '')
                    if key_district.isdigit():
                        continue
                    result = result.split(key_district)[0]
            elif 'Thanh Pho' in district:
                for key in ['Thanh Pho ', 'Thanh Pho', 'Tp ', 'Tp.', ',', '.', '']:
                    key_district = district.replace('Thanh Pho ', key).strip()
                    result = result.split(key_district)[0]

        # city
        if city != None:
            city = city.title()
            for key in ['Tinh ', 'Tinh', 'Thanh Pho ', 'Thanh Pho', 'T.', 'Tp', 'Tp.', ',', '.', '']:
                key_city = (key + city).strip()
                result = result.split(key_city)[0]

        # Normalize
        result = result.strip()
        if result in [None, '']:
            result = None
        else:
            result = result[:-1].strip() if (result[-1] in [',', '.']) else result

        # Fix UnitAdress is FullAddress
        if (result != None) & (district != None) & (city != None):
            have_district = False
            for key_district in [' Huyen ', ' Thi Xa ', ' Quan ', ' Thanh Pho ']:
                if key_district.lower() in result.lower():
                    have_district = True
                    break

            have_city = False
            for key_city in [' Tinh ', ' Thanh Pho ']:
                if key_city.lower() in result.lower():
                    have_city = True
                    break

            if (have_district == True) & (have_city == True):
                result = None

        if (result != None) & (district != None):
            for key_district in [' Huyen ', ' Thi Xa ', ' Quan ', ' Thanh Pho ']:
                if key_district.lower() in result.lower():
                    result = result.split(',')[0].strip()
                    if len(result.split(' ')) > 5:
                        result = None
                    break

        return result

    def UltimatelyUnescape(s: str) -> str:
        unescaped = ""
        while unescaped != s:
            s = html.unescape(s)
            unescaped = html.unescape(s)

        return s

    condition_address = profile_sendo['address'].notna()
    profile_sendo.loc[condition_address, 'address'] = profile_sendo.loc[condition_address, 'address'].str.lower().apply(UltimatelyUnescape).str.title()
    profile_sendo['address'] = profile_sendo['address'].replace({'[0-9]{6,}': ''}, regex=True)

    profile_sendo.loc[condition_address, 'unit_address'] = profile_sendo[condition_address].apply(lambda x: SparseUnitAddress(x.address, x.ward, x.district, x.city), axis=1)
    profile_sendo.loc[profile_sendo['unit_address'].isna(), 'unit_address'] = None
    # profile_sendo['unit_address'] = profile_sendo['unit_address'].str.replace('. ', ', ').str.replace('; ', ',')

    # unify city
    profile_sendo['city'] = profile_sendo['city'].replace({'Ba Ria-Vung Tau': 'Vung Tau', 
                                                           'Dak Nong': 'Dac Nong', 
                                                           'Bac Kan': 'Bac Can'})

    norm_sendo_city = pd.read_parquet('/data/fpt/ftel/cads/dep_solution/user/namdp11/scross_fill/runner/refactor/material/ftel_provinces.parquet', 
                                      filesystem=hdfs)
    norm_sendo_city.columns = ['city', 'norm_city']
    profile_sendo = profile_sendo.merge(norm_sendo_city, how='left', on='city')
    profile_sendo['city'] = profile_sendo['norm_city']
    profile_sendo = profile_sendo.drop(columns=['norm_city'])

    # unify district
    def UnifyDistrictSendo(dict_location, district, city):
        if district == None:
            return None

        district = unidecode(district)
        location = dict_location[['district', 'city']].drop_duplicates().copy()

        if city != None:
            location = location[location['city'] == city]
        if district in ['Quan Thu Duc', 'Quan 9', 'Quan 2', 'Thanh pho Thu Duc']:
            return district

        temp_district = district.title().replace('Huyen ', '').replace('Thi Xa ', '').replace('Thanh Pho ', '').strip()
        location['district_temp'] = location['district'].str.title().replace({'Huyen ': '', 
                                                                              'Thi Xa': '',
                                                                              'Thanh Pho ': '', 
                                                                              'Quan ': ''}, regex=True).str.strip()
        location['similar'] = location['district_temp'].apply(lambda x: SequenceMatcher(None, temp_district.lower(), x.lower()).ratio())

        location = location[(location['similar'] == location['similar'].max()) & 
                            (location['similar'] >= 0.8)]

        unify_district = None
        if location.empty == False:
            unify_district = location['district'].values[0]

        if unify_district == None:
            unify_district = district

        return unify_district

    stats_district_sendo = profile_sendo.groupby(by=['district', 'city'])['id_sendo'].agg(num_customer='count').reset_index()
    dict_district = dict_location[['district', 'city']].drop_duplicates().copy()
    dict_district['new_district'] = dict_district['district']
    stats_district_sendo = stats_district_sendo.merge(dict_district, how='left', on=['district', 'city'])

    condition_district = stats_district_sendo['new_district'].isna()
    stats_district_sendo.loc[condition_district, 'new_district'] = stats_district_sendo[condition_district].apply(lambda x: UnifyDistrictSendo(dict_location, x.district, x.city), axis=1)
    stats_district_sendo = stats_district_sendo.drop(columns=['num_customer'])

    profile_sendo = profile_sendo.merge(stats_district_sendo, how='left', on=['district', 'city'])
    profile_sendo['district'] = profile_sendo['new_district']
    profile_sendo = profile_sendo.drop(columns=['new_district'])

    # unify ward
    def UnifyWardSendo(dict_location, ward, district, city):
        if ward == None:
            return None

        ward = unidecode(ward).title()
        location = dict_location[['ward', 'district', 'city']].drop_duplicates().copy()

        if city != None:
            location = location[location['city'] == city]
        if district != None:
            if district in ['Quan Thu Duc', 'Quan 9', 'Quan 2']:
                district = 'Thanh pho Thu Duc'
            location = location[location['district'] == district]

        temp_ward = ward.title().replace('Xa ', '').replace('Phuong ', '').replace('Thi Tran ', '').replace('0', '').strip()
        location['ward_temp'] = location['ward'].str.title().replace({'Xa ': '',
                                                                      'Phuong ': '',
                                                                      'Thi Tran ': '',
                                                                      '0': ''}, regex=True).str.strip()
        location['similar'] = location['ward_temp'].apply(lambda x: SequenceMatcher(None, temp_ward.lower(), x.lower()).ratio())

        location = location[(location['similar'] == location['similar'].max()) & 
                            (location['similar'] >= 0.8)]

        unify_ward = None
        if location.empty == False:
            unify_ward = location['ward'].values[0]

        if unify_ward == None:
            unify_ward = ward

        return unify_ward

    stats_ward_sendo = profile_sendo.groupby(by=['ward', 'district', 'city'])['id_sendo'].agg(num_customer='count').reset_index()

    dict_ward = dict_location[['ward', 'district', 'city']].drop_duplicates().copy()
    dict_ward['new_ward'] = dict_ward['ward']
    stats_ward_sendo = stats_ward_sendo.merge(dict_ward, how='left', on=['ward', 'district', 'city'])

    condition_ward = stats_ward_sendo['new_ward'].isna()
    stats_ward_sendo.loc[condition_ward, 'new_ward'] = stats_ward_sendo[condition_ward].apply(lambda x: UnifyWardSendo(dict_location, x.ward, x.district, x.city), axis=1)
    stats_ward_sendo = stats_ward_sendo.drop(columns=['num_customer'])

    profile_sendo = profile_sendo.merge(stats_ward_sendo, how='left', on=['ward', 'district', 'city'])
    profile_sendo['ward'] = profile_sendo['new_ward']
    profile_sendo = profile_sendo.drop(columns=['new_ward'])
    
    condition_ward_error = profile_sendo['ward'].str.contains('0[1-9]', case=False, na=False)
    profile_sendo.loc[condition_ward_error, 'ward'] = profile_sendo.loc[condition_ward_error, 'ward'].str.replace('0', '')

    # full_address
    profile_sendo['address'] = None
    columns = ['unit_address', 'ward', 'district', 'city']
    profile_sendo['address'] = profile_sendo[columns].fillna('').agg(', '.join, axis=1).str.replace('(?<![a-zA-Z0-9]),', '', regex=True).str.replace('-(?![a-zA-Z0-9])', '', regex=True)

    # add info
    profile_sendo['gender'] = None
    profile_sendo['birthday'] = None
    columns = ['id_sendo', 'phone_raw', 'phone', 'is_phone_valid', 
               'email_raw', 'email', 'is_email_valid', 
               'name', 'pronoun', 'is_full_name', 'gender', 
               'birthday', 'customer_type', # 'customer_type_detail',
               'address', 'unit_address', 'ward', 'district', 'city', 'source']
    profile_sendo = profile_sendo[columns]

    # Fill 'Ca nhan'
    profile_sendo.loc[profile_sendo['name'].notna() & profile_sendo['customer_type'].isna(), 'customer_type'] = 'Ca nhan'
    
    # Create id_phone_sendo
    profile_sendo.loc[profile_sendo['id_sendo'].notna() & 
                  profile_sendo['phone'].notna(), 'id_phone_sendo'] = profile_sendo['id_sendo'] + '-' + profile_sendo['phone']
    profile_sendo.loc[profile_sendo['id_sendo'].notna() & 
                      profile_sendo['phone'].isna(), 'id_phone_sendo'] = profile_sendo['id_sendo']

    # Save
    profile_sendo['d'] = date_str
    profile_sendo.drop_duplicates().to_parquet(f'{unify_path}/sendo.parquet', 
                             filesystem=hdfs,
                             index=False,
                             partition_cols='d')
    
if __name__ == '__main__':
    
    date_str = sys.argv[1]
    UnifySendo(date_str)
