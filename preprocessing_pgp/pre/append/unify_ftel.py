
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


def UnifyFtel(profile_ftel: pd.DataFrame):
    # VARIABLE
    dict_trash = {'': None, 'Nan': None, 'nan': None, 'None': None,
                  'none': None, 'Null': None, 'null': None, "''": None}
    dict_location = pd.read_parquet('/data/fpt/ftel/cads/dep_solution/user/namdp11/scross_fill/runner/refactor/material/dict_location.parquet',
                                    filesystem=hdfs)

    print(">>> Cleansing profile")
    condition_name = profile_ftel['name'].notna()
    profile_ftel.loc[condition_name, 'name'] =\
        profile_ftel.loc[condition_name, 'name']\
        .apply(basic_preprocess_name)
    profile_ftel.rename(columns={
        'email': 'email_raw',
        'phone': 'phone_raw',
        'name': 'raw_name'
    }, inplace=True)

    # * Loadding dictionary
    print(">>> Loading dictionaries")
    profile_phones = profile_ftel['phone_raw'].drop_duplicates().dropna()
    profile_emails = profile_ftel['email_raw'].drop_duplicates().dropna()
    profile_names = profile_ftel['raw_name'].drop_duplicates().dropna()

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
    profile_ftel = profile_ftel.rename(columns={'contract_ftel': 'contract'})

    # merge get phone, email (valid)
    print(">>> Merging phone, email, name")
    profile_ftel = pd.merge(
        profile_ftel.set_index('phone_raw'),
        valid_phone.set_index('phone_raw'),
        left_index=True, right_index=True,
        how='left',
        sort=False
    ).reset_index(drop=False)

    profile_ftel = pd.merge(
        profile_ftel.set_index('email_raw'),
        valid_email.set_index('email_raw'),
        left_index=True, right_index=True,
        how='left',
        sort=False
    ).reset_index(drop=False)

    profile_ftel = pd.merge(
        profile_ftel.set_index('raw_name'),
        dict_name_lst.set_index('raw_name'),
        left_index=True, right_index=True,
        how='left',
        sort=False
    ).rename(columns={
        'enrich_name': 'name'
    }).reset_index(drop=False)

    # load datapay => customer type
    # ds_contract = pd.read_parquet('/data/fpt/ftel/isc/dwh/ds_contract.parquet',
    #                               columns=['contract', 'net_customer_type'],
    #                               filesystem=hdfs).drop_duplicates(subset=['contract'], keep='last')
    # ds_contract.columns = ['contract', 'datapay_customer_type']
    # profile_ftel = profile_ftel.merge(ds_contract, how='left', on='contract')
    # profile_ftel.loc[profile_ftel['source'] ==
    #                  'multi', 'datapay_customer_type'] = None
    # profile_ftel.loc[profile_ftel['source'] == 'multi', 'city'] = None

    # customer type
    # profile_ftel.loc[profile_ftel['customer_type'].isna(
    # ), 'customer_type'] = profile_ftel['datapay_customer_type']
    # profile_ftel = profile_ftel.drop(columns=['datapay_customer_type'])

    # birthday
    print(">>> Processing Birthday")
    condition_birthday = profile_ftel['birthday'].notna()
    profile_ftel.loc[condition_birthday, 'birthday'] =\
        pd.to_datetime(
            profile_ftel[condition_birthday]['birthday'].astype(str),
            errors='coerce'
    ).dt.strftime('%d/%m/%Y')

    # drop name is username_email
    print(">>> Extra Cleansing Name")
    profile_ftel['username_email'] = profile_ftel['email'].str.split(
        '@').str[0]
    profile_ftel.loc[profile_ftel['name'] ==
                     profile_ftel['username_email'], 'name'] = None
    profile_ftel = profile_ftel.drop(columns=['username_email'])

    # clean name
    name_process = NameProcess()
    condition_name = (profile_ftel['customer_type'] == 'customer')\
        & (profile_ftel['name'].notna())

    profile_ftel.loc[
        condition_name,
        ['clean_name', 'pronoun']
    ] = profile_ftel.loc[
        condition_name,
        'name'
    ].apply(name_process.CleanName).tolist()

    profile_ftel.loc[
        profile_ftel['customer_type'] == 'customer',
        'name'
    ] = profile_ftel['clean_name']
    profile_ftel = profile_ftel.drop(columns=['clean_name'])

    # skip pronoun
    profile_ftel['name'] = profile_ftel['name'].str.strip().str.title()
    skip_names = ['Vợ', 'Vo', 'Anh', 'Chị', 'Chi', 'Mẹ', 'Me', 'Em', 'Ba',
                  'Chú', 'Chu', 'Bác', 'Bac', 'Ông', 'Ong', 'Cô', 'Co', 'Cha', 'Dì', 'Dượng']
    profile_ftel.loc[profile_ftel['name'].isin(skip_names), 'name'] = None

    # is full name
    print(">>> Checking Full Name")
    profile_ftel.loc[profile_ftel['last_name'].notna(
    ) & profile_ftel['first_name'].notna(), 'is_full_name'] = True
    profile_ftel['is_full_name'] = profile_ftel['is_full_name'].fillna(False)
    profile_ftel = profile_ftel.drop(
        columns=['last_name', 'middle_name', 'first_name'])
    profile_ftel['name'] = profile_ftel['name'].str.strip().str.title()

    # Transfering model gender
    print(">>> Transfering Model Gender")
    profile_ftel.loc[
        profile_ftel['customer_type'] != 'customer',
        'gender'
    ] = None
    profile_ftel.loc[
        profile_ftel['gender_enrich'].notna(),
        'gender'
    ] = profile_ftel.loc[
        profile_ftel['gender_enrich'].notna(),
        'gender_enrich'
    ]

    # unify location
    print(">>> Processing Address")
    norm_ftel_city = pd.read_parquet('/data/fpt/ftel/cads/dep_solution/user/namdp11/scross_fill/runner/refactor/material/ftel_provinces.parquet',
                                     filesystem=hdfs)
    norm_ftel_city.columns = ['city', 'norm_city']

    norm_ftel_district = pd.read_parquet('/data/fpt/ftel/cads/dep_solution/user/namdp11/scross_fill/runner/refactor/material/ftel_districts.parquet',
                                         filesystem=hdfs)
    norm_ftel_district.columns = [
        'district', 'norm_city', 'norm_district', 'new_norm_district']

    # update miss district
    district_update = list(
        set(profile_ftel['district']) - set(norm_ftel_district['district']))
    location_dict = pd.read_parquet('/data/fpt/ftel/cads/dep_solution/user/namdp11/scross_fill/runner/refactor/material/location_dict.parquet',
                                    filesystem=hdfs)
    location_dict.columns = ['city', 'district', 'norm_city', 'norm_district']
    district_list = list(location_dict['norm_district'].unique())

    def fix_miss_district(district_list, district):
        df = pd.DataFrame()
        df['district'] = district_list
        df['similar'] = df['district'].apply(
            lambda x: SequenceMatcher(None, district.lower(), x.lower()).ratio())

        df = df[df.similar == df.similar.max()]
        result = df['district'].values[0]
        return result

    miss_district = {}
    for district in district_update:
        if district != None:
            result = fix_miss_district(district_list, district)
            miss_district[district] = result

    # merge city, district
    profile_ftel = pd.merge(
        profile_ftel.set_index('city'),
        norm_ftel_city.set_index('city'),
        left_index=True, right_index=True,
        how='left',
        sort=False
    )
    profile_ftel = profile_ftel.merge(
        norm_ftel_district[['district', 'norm_city', 'new_norm_district']],
        how='left', on=['district', 'norm_city']
    )

    # fix bugs location
    profile_ftel.loc[profile_ftel['contract'].notna() &
                     profile_ftel['phone'].notna(), 'contract_phone_ftel'] = profile_ftel['phone'] + '-' + profile_ftel['contract']
    profile_ftel.loc[profile_ftel['contract'].notna() &
                     profile_ftel['phone'].isna(), 'contract_phone_ftel'] = profile_ftel['contract']

    bug_location_ftel = profile_ftel[profile_ftel['norm_city'].isna(
    ) | profile_ftel['new_norm_district'].isna()]
    bug_location_ftel1 = bug_location_ftel[bug_location_ftel['norm_city'].notna(
    )]
    bug_location_ftel2 = bug_location_ftel[~bug_location_ftel['contract_phone_ftel'].isin(
        bug_location_ftel1['contract_phone_ftel'])]

    bug_location_ftel1 = bug_location_ftel1.drop(columns=['new_norm_district'])
    bug_location_ftel1 = bug_location_ftel1.merge(
        norm_ftel_district[['district', 'new_norm_district']],
        how='left', on=['district']
    )

    bug_location_ftel2 = bug_location_ftel2.drop(
        columns=['norm_city', 'new_norm_district'])
    bug_location_ftel2 = bug_location_ftel2.merge(
        norm_ftel_district[['district', 'norm_city', 'new_norm_district']],
        how='left', on=['district']
    )

    bug_location_ftel = bug_location_ftel1.append(
        bug_location_ftel2, ignore_index=True)

    profile_ftel = profile_ftel[~profile_ftel['contract_phone_ftel'].isin(
        bug_location_ftel['contract_phone_ftel'])]
    profile_ftel = profile_ftel.append(bug_location_ftel, ignore_index=True)

    profile_ftel.loc[profile_ftel['district'].isin(miss_district.keys(
    )), 'new_norm_district'] = profile_ftel['district'].map(miss_district)

    profile_ftel['city'] = profile_ftel['norm_city']
    profile_ftel['district'] = profile_ftel['new_norm_district']
    profile_ftel = profile_ftel.drop(
        columns=['norm_city', 'new_norm_district'])

    # fix distric-city
    dict_district = dict_location[[
        'district', 'city']].drop_duplicates().copy()
    stats_district = dict_district.groupby(
        by=['district'])['city'].agg(num_city='count').reset_index()
    stats_district = stats_district[stats_district['num_city'] == 1]
    dict_district = dict_district[dict_district['district'].isin(
        stats_district['district'])]
    dict_district = dict_district.rename(columns={'city': 'new_city'})
    dict_district = dict_district.append([{'district': 'Quan Thu Duc', 'new_city': 'Thanh pho Ho Chi Minh'},
                                          {'district': 'Quan 9',
                                              'new_city': 'Thanh pho Ho Chi Minh'},
                                          {'district': 'Quan 2', 'new_city': 'Thanh pho Ho Chi Minh'}],
                                         ignore_index=True)

    profile_ftel = profile_ftel.merge(dict_district, how='left', on='district')
    profile_ftel.loc[profile_ftel['new_city'].notna(
    ), 'city'] = profile_ftel['new_city']
    profile_ftel = profile_ftel.drop(columns=['new_city'])

    # unify ward
    def UnifyWardFTel1(dict_location, ward, district, city):
        if ward == None:
            return None

        ward = unidecode(ward)
        location = dict_location[['ward', 'district',
                                  'city']].drop_duplicates().copy()

        if city != None:
            location = location[location['city'] == city]
        if district != None:
            if district in ['Quan Thu Duc', 'Quan 9', 'Quan 2']:
                district = 'Thanh pho Thu Duc'
            location = location[location['district'] == district]

        location['ward_temp'] = location['ward'].str.replace(
            'Phuong 0', 'Phuong ')

        location['similar'] = location['ward_temp'].apply(
            lambda x: SequenceMatcher(None, ward.lower(), x.lower()).ratio())

        location = location[(location['similar'] == location['similar'].max()) &
                            (location['similar'] >= 0.8)]

        unify_ward = None
        if location.empty == False:
            unify_ward = location['ward'].values[0]

        return unify_ward

    def UnifyWardFTel2(dict_location, ward, district):
        if ward == None:
            return None

        ward = unidecode(ward)
        location = dict_location[['ward', 'district']].drop_duplicates().copy()

        ward = ward.title().replace('P.', 'Phuong ').replace('F.', 'Phuong ')
        ward = ward.title().replace('T.Tran', 'Thi Tran ').replace(
            'T.T', 'Thi Tran ').replace('Tt', 'Thi Tran ')
        ward = ward.title().replace('Tx', 'Thi Xa ')

        if district != None:
            location = location[location['district'] == district]

        location['ward_temp'] = location['ward'].str.replace(
            'Phuong 0', 'Phuong ')

        location['similar'] = location['ward_temp'].apply(
            lambda x: SequenceMatcher(None, ward.lower(), x.lower()).ratio())

        location = location[(location['similar'] == location['similar'].max()) &
                            (location['similar'] >= 0.8)]

        unify_ward = None
        if location.empty == False:
            unify_ward = location['ward'].values[0]

        return unify_ward

    def UnifyWardFTel3(dict_location, ward, district):
        if ward == None:
            return None

        ward = unidecode(ward)
        location = dict_location[['ward', 'district']].drop_duplicates().copy()

        if district != None:
            location = location[location['district'] == district]

        ward = ward.title().replace('P.', '').replace('F.', '')
        ward = ward.title().replace('T.Tran', '').replace('T.T', '').replace('Tt', '')
        ward = ward.title().replace('Tx', '')

        location['ward_temp'] = location['ward'].str.replace(
            'Phuong 0', 'Phuong ')
        location['ward_temp'] = location['ward_temp'].replace({'Phuong ': '',
                                                               'Xa ': '',
                                                               'Thi Tran ': '',
                                                               'Thi Xa ': ''}, regex=True)

        location['similar'] = location['ward_temp'].apply(
            lambda x: SequenceMatcher(None, ward.lower(), x.lower()).ratio())

        location = location[(location['similar'] == location['similar'].max()) &
                            (location['similar'] >= 0.8)]

        unify_ward = None
        if location.empty == False:
            unify_ward = location['ward'].values[0]

        return unify_ward

    def UnifyWardFTel4(ward):
        if ward == None:
            return None

        ward = unidecode(ward)
        ward = ward.title()
        unify_ward = None
        for key in ['Xa', 'Phuong', 'Huyen', 'Thi Tran', 'T.Tran', 'Tt', 'T.T', 'P.', 'F.', 'Thi Xa', 'Tx']:
            check_idx = ward.find(key)
            if check_idx == 0:
                if key in ['Xa', 'Phuong', 'Huyen']:
                    unify_ward = ward
                elif key in ['Thi Tran', 'T.Tran', 'Tt', 'T.T']:
                    unify_ward = ward.replace(key, 'Thi tran')
                elif key in ['P.', 'F.']:
                    unify_ward = ward.replace(key, 'Phuong ')
                elif key in ['Thi Xa', 'Tx']:
                    unify_ward = ward.replace(key, 'Thi xa')

                break

        if unify_ward != None:
            if len(unify_ward.split(' ')) < 2:
                unify_ward = None

        return unify_ward

    def UnifyWardFTel(dict_location, ward, district, city):
        unify_ward = UnifyWardFTel1(dict_location, ward, district, city)

        if unify_ward == None:
            unify_ward = UnifyWardFTel2(dict_location, ward, district)

        if unify_ward == None:
            unify_ward = UnifyWardFTel3(dict_location, ward, district)

        if unify_ward == None:
            unify_ward = UnifyWardFTel4(ward)

        return unify_ward

    stats_ward = profile_ftel.groupby(by=['ward', 'district', 'city'],
                                      dropna=False)['contract'].agg(num_customer='count').reset_index()
    stats_ward.loc[stats_ward['ward'].isna(), 'ward'] = None
    stats_ward.loc[stats_ward['district'].isna(), 'district'] = None
    stats_ward.loc[stats_ward['city'].isna(), 'city'] = None

    stats_ward['unify_ward'] = stats_ward.apply(lambda x:
                                                UnifyWardFTel(
                                                    dict_location, x.ward, x.district, x.city),
                                                axis=1)

    profile_ftel = profile_ftel.merge(
        stats_ward[['ward', 'district', 'city', 'unify_ward']],
        how='left', on=['ward', 'district', 'city']
    )
    profile_ftel['ward'] = profile_ftel['unify_ward']
    profile_ftel = profile_ftel.drop(columns=['unify_ward'])

    # unit_address
    columns = ['house_number', 'street']
    profile_ftel['unit_address'] = profile_ftel[columns].fillna('').agg(' '.join, axis=1).str.replace(
        '(?<![a-zA-Z0-9]),', '', regex=True).str.replace('-(?![a-zA-Z0-9])', '', regex=True)
    profile_ftel['unit_address'] = profile_ftel['unit_address'].str.strip(
    ).replace(dict_trash)
    profile_ftel['unit_address'] = profile_ftel['unit_address'].str.title()

    # full_address
    columns = ['unit_address', 'ward', 'district', 'city']
    profile_ftel['address'] = None
    profile_ftel['address'] = profile_ftel[columns].fillna('').agg(', '.join, axis=1).str.replace(
        '(?<![a-zA-Z0-9]),', '', regex=True).str.replace('-(?![a-zA-Z0-9])', '', regex=True)
    profile_ftel['address'] = profile_ftel['address'].str.strip().replace(
        dict_trash)
    profile_ftel = profile_ftel.drop(columns=['house_number', 'street'])

    # add info
    print(">>> Filtering out info")
    columns = ['contract', 'phone_raw', 'phone', 'is_phone_valid',
               'email_raw', 'email', 'is_email_valid',
               'name', 'pronoun', 'is_full_name', 'gender',
               'birthday', 'customer_type',  # 'customer_type_detail',
               'address', 'unit_address', 'ward', 'district', 'city', 'source']
    profile_ftel = profile_ftel[columns]
    profile_ftel = profile_ftel.rename(columns={'contract': 'contract_ftel'})

    # Create contract_phone_ftel
    profile_ftel.loc[
        (profile_ftel['contract_ftel'].notna())
        & (profile_ftel['phone'].notna()),
        'contract_phone_ftel'
    ] = profile_ftel['contract_ftel'] + '-' + profile_ftel['phone']
    profile_ftel.loc[
        (profile_ftel['contract_ftel'].notna())
        & (profile_ftel['phone'].isna()),
        'contract_phone_ftel'
    ] = profile_ftel['contract_ftel']

    # return
    return profile_ftel

# function update profile (unify)


def UpdateUnifyFtel(now_str):
    # VARIABLES
    raw_path = ROOT_PATH + '/raw'
    unify_path = ROOT_PATH + '/pre'
    f_group = 'ftel'
    yesterday_str = (datetime.strptime(now_str, '%Y-%m-%d') -
                     timedelta(days=1)).strftime('%Y-%m-%d')

    # load profile (yesterday, now)
    print(">>> Loading today and yesterday profile")
    info_columns = ['contract_ftel', 'phone', 'email', 'name', 'birthday',
                    'address', 'house_number', 'street', 'ward', 'district', 'city', 'source']
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
        new_profile_unify = UnifyFtel(difference_profile)

        # synthetic profile
        profile_unify = pd.concat(
            [new_profile_unify, profile_unify],
            ignore_index=True
        )

    # arrange columns
    columns = [
        'contract_ftel', 'phone_raw', 'phone', 'is_phone_valid',
        'email_raw', 'email', 'is_email_valid',
        'name', 'pronoun', 'is_full_name', 'gender',
        'birthday', 'customer_type',  # 'customer_type_detail',
        'address', 'unit_address', 'ward', 'district', 'city',
        'source', 'contract_phone_ftel'
    ]
    profile_unify = profile_unify[columns].copy()
    profile_unify['birthday'] = profile_unify['birthday'].astype(str)
    profile_unify = profile_unify.drop_duplicates(
        subset=['contract_ftel', 'phone_raw', 'email_raw'], keep='first')

    # skip name (multi)
    profile_unify.loc[profile_unify['source'] == 'multi', 'name'] = None

    # save
    profile_unify['d'] = now_str
    profile_unify.to_parquet(
        f'{unify_path}/{f_group}.parquet',
        filesystem=hdfs, index=False,
        partition_cols='d'
    )


if __name__ == '__main__':

    now_str = sys.argv[1]
    UpdateUnifyFtel(now_str)
