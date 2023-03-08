
import pandas as pd
import numpy as np
import multiprocessing as mp
import sys

import os
import subprocess
from pyarrow import fs

from preprocessing_pgp.name.type.extractor import process_extract_name_type

os.environ['HADOOP_CONF_DIR'] = "/etc/hadoop/conf/"
os.environ['JAVA_HOME'] = "/usr/jdk64/jdk1.8.0_112"
os.environ['HADOOP_HOME'] = "/usr/hdp/3.1.0.0-78/hadoop"
os.environ['ARROW_LIBHDFS_DIR'] = "/usr/hdp/3.1.0.0-78/usr/lib/"
os.environ['CLASSPATH'] = subprocess.check_output("$HADOOP_HOME/bin/hadoop classpath --glob", shell=True).decode('utf-8')
hdfs = fs.HadoopFileSystem(host="hdfs://hdfs-cluster.datalake.bigdata.local", port=8020)

sys.path.append('/bigdata/fdp/cdp/source/core_profile/preprocess/utils')
from preprocess_profile import (
    cleansing_profile_name,
    extracting_pronoun_from_name
)
from const import (
    UTILS_PATH,
    CENTRALIZE_PATH,
    PREPROCESS_PATH
)

def UnifyCredit(
    date_str: str,
    n_cores: int = 1
):
    # VARIABLES
    f_group = 'frt_credit'

    dict_trash = {'': None, 'Nan': None, 'nan': None, 'None': None, 'none': None, 'Null': None, 'null': None, "''": None}
    # load profile credit
    profile_credit = pd.read_parquet(f'{CENTRALIZE_PATH}/{f_group}.parquet/d={date_str}', filesystem=hdfs)
    # * Cleansing
    print(">>> Cleansing profile")
    profile_credit = cleansing_profile_name(
        profile_credit,
        name_col='name',
        n_cores=n_cores
    )
    profile_credit.rename(columns={
        'phone': 'phone_raw',
        'name': 'raw_name'
    }, inplace=True)

    # * Loading dictionary
    print(">>> Loading dictionaries")
    profile_phones = profile_credit['phone_raw'].drop_duplicates().dropna()
    # profile_emails = profile_credit['email_raw'].drop_duplicates().dropna()
    profile_names = profile_credit['raw_name'].drop_duplicates().dropna()

    # phone(valid)
    valid_phone = pd.read_parquet(
        f'{UTILS_PATH}/valid_phone_latest.parquet',
        filters=[('phone_raw', 'in', profile_phones)],
        filesystem=hdfs,
        columns=['phone_raw', 'phone', 'is_phone_valid']
    )
    # valid_email = pd.read_parquet(
    #     f'{UTILS_PATH}/valid_email_latest.parquet',
    #     filters=[('email_raw', 'in', profile_emails)],
    #     filesystem=hdfs,
    #     columns=['email_raw', 'email', 'is_email_valid']
    # )
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
    profile_credit = profile_credit[['uid', 'phone', 'name', 'gender', 'birthday', 'address', 'customer_type']]
    profile_credit['email'] = None

    profile_credit = profile_credit.rename(
        columns={
            'customer_type': 'customer_type_credit',
        }
    )
    profile_credit.loc[profile_credit['gender'] == '-1', 'gender'] = None
    profile_credit.loc[profile_credit['address'].isin(
        ['', 'Null', 'None', 'Test']), 'address'] = None
    profile_credit.loc[profile_credit['address'].notna(
    ) & profile_credit['address'].str.isnumeric(), 'address'] = None
    profile_credit.loc[profile_credit['address'].str.len() <
                       5, 'address'] = None
    profile_credit['customer_type_credit'] =\
        profile_credit['customer_type_credit']\
        .replace({
            'Individual': 'Ca nhan',
            'Company': 'Cong ty',
            'Other': None
        })

    # merge get phone(valid)
    print(">>> Merging phone, email, name")
    profile_credit = pd.merge(
        profile_credit.set_index('phone_raw'),
        valid_phone.set_index('phone_raw'),
        left_index=True, right_index=True,
        how='left',
        sort=False
    ).reset_index(drop=False)

    profile_credit = pd.merge(
        profile_credit.set_index('raw_name'),
        dict_name_lst.set_index('raw_name'),
        left_index=True, right_index=True,
        how='left',
        sort=False
    ).rename(columns={
        'enrich_name': 'name'
    }).reset_index(drop=False)

    # Refilling info
    cant_predict_name_mask = profile_credit['name'].isna()
    profile_credit.loc[
        cant_predict_name_mask,
        'name'
    ] = profile_credit.loc[
        cant_predict_name_mask,
        'raw_name'
    ]
    profile_credit['name'] = profile_credit['name'].replace(dict_trash)

    # customer_type
    print(">>> Processing Customer Type")
    profile_credit = process_extract_name_type(
        profile_credit,
        name_col='name',
        n_cores=n_cores,
        logging_info=False
    )
    profile_credit['customer_type'] =\
        profile_credit['customer_type'].map({
            'customer': 'Ca nhan',
            'company': 'Cong ty',
            'medical': 'Benh vien - Phong kham',
            'edu': 'Giao duc',
            'biz': 'Ho kinh doanh'
        })
    profile_credit.loc[
        profile_credit['customer_type'] == 'Ca nhan',
        'customer_type'
    ] = profile_credit['customer_type_credit']
    profile_credit = profile_credit.drop(columns=['customer_type_credit'])

    # clean name
    condition_name =\
        (profile_credit['customer_type'].isin([None, 'Ca nhan', np.nan]))\
        & (profile_credit['name'].notna())
    profile_credit = extracting_pronoun_from_name(
        profile_credit,
        condition=condition_name,
        name_col='name',
    )

    # is full name
    print(">>> Checking Full Name")
    profile_credit.loc[profile_credit['last_name'].notna(
    ) & profile_credit['first_name'].notna(), 'is_full_name'] = True
    profile_credit['is_full_name'] = profile_credit['is_full_name'].fillna(
        False)
    profile_credit = profile_credit.drop(
        columns=['last_name', 'middle_name', 'first_name'])

    # valid gender by model
    print(">>> Validating Gender")
    profile_credit.loc[
        profile_credit['customer_type'] != 'Ca nhan',
        'gender'
    ] = None

    profile_credit.loc[
        (profile_credit['gender'].notna())
        & (profile_credit['gender'] != profile_credit['gender_enrich']),
        'gender'
    ] = None

    # normlize address
    print(">>> Processing Address")
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
    print(">>> Adding Temp Info")
    columns = ['uid', 'phone_raw', 'phone', 'is_phone_valid', 
               # 'email_raw', 'email', 'is_email_valid',
               'name', 'pronoun', 'is_full_name', 'gender', 
               'birthday', 'customer_type', # 'customer_type_detail',
               'address', 'source_address', 'unit_address', 'source_unit_address', 
               'ward', 'source_ward', 'district', 'source_district', 'city', 'source_city']
    profile_credit = profile_credit[columns]

    # Fill 'Ca nhan'
    profile_credit.loc[
        (profile_credit['name'].notna())
        & (profile_credit['customer_type'].isna()),
        'customer_type'
    ] = 'Ca nhan'

    # Save
    profile_credit['d'] = date_str
    profile_credit.to_parquet(
        f'{PREPROCESS_PATH}/{f_group}.parquet',
        filesystem=hdfs, index=False,
        partition_cols='d'
    )

if __name__ == '__main__':

    DAY = sys.argv[1]
    UnifyCredit(DAY, n_cores=5)
