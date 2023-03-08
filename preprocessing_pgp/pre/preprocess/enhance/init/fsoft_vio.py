
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
    remove_same_username_email,
    cleansing_profile_name,
    extracting_pronoun_from_name
)
from const import (
    UTILS_PATH,
    CENTRALIZE_PATH,
    PREPROCESS_PATH
)

def UnifyFsoft(
    date_str: str,
    n_cores: int = 1
):
    # VARIABLES
    f_group = 'fsoft'

    dict_trash = {'': None, 'Nan': None, 'nan': None, 'None': None,
                  'none': None, 'Null': None, 'null': None, "''": None}

    # load profile fsoft
    profile_fsoft = pd.read_parquet(f'{CENTRALIZE_PATH}/{f_group}.parquet/d={date_str}', filesystem=hdfs)

    # * Cleansing
    print(">>> Cleansing profile")
    profile_fsoft = cleansing_profile_name(
        profile_fsoft,
        name_col='name',
        n_cores=n_cores
    )
    profile_fsoft.rename(columns={
        'email': 'email_raw',
        'phone': 'phone_raw',
        'name': 'raw_name'
    }, inplace=True)

    # * Loading dictionary
    print(">>> Loading dictionaries")
    profile_phones = profile_fsoft['phone_raw'].drop_duplicates().dropna()
    profile_emails = profile_fsoft['email_raw'].drop_duplicates().dropna()
    profile_names = profile_fsoft['raw_name'].drop_duplicates().dropna()

    # phone, email(valid)
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
    profile_fsoft = profile_fsoft[['uid', 'phone', 'email', 'name', 'birthday', 'address', 'city', 'district', 'customer_type']]
    profile_fsoft = profile_fsoft.rename(columns={'customer_type': 'customer_type_fsoft', 'phone': 'phone_raw', 'email': 'email_raw'})
    # profile_fsoft.loc[profile_fsoft['gender'] == '-1', 'gender'] = None
    profile_fsoft.loc[profile_fsoft['address'].isin(
        ['', 'Null', 'None', 'Test']), 'address'] = None
    profile_fsoft.loc[profile_fsoft['address'].notna(
    ) & profile_fsoft['address'].str.isnumeric(), 'address'] = None
    profile_fsoft.loc[profile_fsoft['address'].str.len() < 5, 'address'] = None
    profile_fsoft['customer_type_fsoft'] =\
        profile_fsoft['customer_type_fsoft']\
        .replace({
            'Individual': 'Ca nhan',
            'Company': 'Cong ty',
            'Other': None
        })

    # merge get phone, email (valid) and names
    print(">>> Merging phone, email, name")
    profile_fsoft = pd.merge(
        profile_fsoft.set_index('phone_raw'),
        valid_phone.set_index('phone_raw'),
        left_index=True, right_index=True,
        how='left',
        sort=False
    ).reset_index(drop=False)

    profile_fsoft = pd.merge(
        profile_fsoft.set_index('email_raw'),
        valid_email.set_index('email_raw'),
        left_index=True, right_index=True,
        how='left',
        sort=False
    ).reset_index(drop=False)

    profile_fsoft = pd.merge(
        profile_fsoft.set_index('raw_name'),
        dict_name_lst.set_index('raw_name'),
        left_index=True, right_index=True,
        how='left',
        sort=False
    ).rename(columns={
        'enrich_name': 'name'
    }).reset_index(drop=False)

    # Refilling info
    cant_predict_name_mask = profile_fsoft['name'].isna()
    profile_fsoft.loc[
        cant_predict_name_mask,
        'name'
    ] = profile_fsoft.loc[
        cant_predict_name_mask,
        'raw_name'
    ]
    profile_fsoft['name'] = profile_fsoft['name'].replace(dict_trash)

    # customer_type
    print(">>> Processing Customer Type")
    profile_fsoft = process_extract_name_type(
        profile_fsoft,
        name_col='name',
        n_cores=n_cores,
        logging_info=False
    )
    profile_fsoft['customer_type'] =\
        profile_fsoft['customer_type'].map({
            'customer': 'Ca nhan',
            'company': 'Cong ty',
            'medical': 'Benh vien - Phong kham',
            'edu': 'Giao duc',
            'biz': 'Ho kinh doanh'
        })
    profile_fsoft.loc[
        profile_fsoft['customer_type'] == 'Ca nhan',
        'customer_type'
    ] = profile_fsoft['customer_type_fsoft']
    profile_fsoft = profile_fsoft.drop(columns=['customer_type_fsoft'])

    # drop name is username_email
    print(">>> Extra Cleansing Name")
    profile_fsoft = remove_same_username_email(
        profile_fsoft,
        name_col='name',
        email_col='email'
    )

    # clean name, extract pronoun
    condition_name = (profile_fsoft['customer_type'] == 'Ca nhan')\
        & (profile_fsoft['name'].notna())
    profile_fsoft = extracting_pronoun_from_name(
        profile_fsoft,
        condition=condition_name,
        name_col='name',
    )

    # is full name
    print(">>> Checking Full Name")
    profile_fsoft.loc[profile_fsoft['last_name'].notna(
    ) & profile_fsoft['first_name'].notna(), 'is_full_name'] = True
    profile_fsoft['is_full_name'] = profile_fsoft['is_full_name'].fillna(False)
    profile_fsoft = profile_fsoft.drop(
        columns=['last_name', 'middle_name', 'first_name'])

    # valid gender by model
    print(">>> Validating Gender")
    profile_fsoft.loc[
        profile_fsoft['customer_type'] != 'Ca nhan',
        'gender'
    ] = None
    # profile_fo.loc[profile_fo['gender'].notna() & profile_fo['name'].isna(), 'gender'] = None
    profile_fsoft.loc[
        (profile_fsoft['gender'].notna())
        & (profile_fsoft['gender'] != profile_fsoft['gender_enrich']),
        'gender'
    ] = None

    # normlize address
    print(">>> Processing Address")
    profile_fsoft['address'] = profile_fsoft['address'].str.strip().replace(
        dict_trash)
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
    columns = ['uid', 'phone_raw', 'phone', 'is_phone_valid',
               'email_raw', 'email', 'is_email_valid',
               'name', 'pronoun', 'is_full_name', 'gender',
               'birthday', 'customer_type', # 'customer_type_detail',
               'address', 'source_address', 'unit_address', 'source_unit_address',
               'ward', 'source_ward', 'district', 'source_district', 'city', 'source_city']
    profile_fsoft = profile_fsoft[columns]

    # Fill 'Ca nhan'
    profile_fsoft.loc[
        (profile_fsoft['name'].notna())
        & (profile_fsoft['customer_type'].isna()),
        'customer_type'
    ] = 'Ca nhan'

    # Save
    profile_fsoft['d'] = date_str
    profile_fsoft.to_parquet(
        f'{PREPROCESS_PATH}/{f_group}.parquet',
        filesystem=hdfs, index=False,
        partition_cols='d'
    )

if __name__ == '__main__':

    DAY = sys.argv[1]
    UnifyFsoft(DAY, n_cores=10)
