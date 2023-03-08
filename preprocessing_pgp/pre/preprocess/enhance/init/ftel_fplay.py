
import pandas as pd
from datetime import datetime, timedelta
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

def UnifyFplay(
    date_str: str,
    n_cores: int = 1
):
    # VARIABLES
    f_group = 'ftel_fplay'

    dict_trash = {'': None, 'Nan': None, 'nan': None, 'None': None, 'none': None, 'Null': None, 'null': None, "''": None}
    # load profile fplay
    profile_fplay = pd.read_parquet(f'{CENTRALIZE_PATH}/{f_group}.parquet/d={date_str}', filesystem=hdfs)

    print(">>> Cleansing profile")
    profile_fplay = cleansing_profile_name(
        profile_fplay,
        name_col='name',
        n_cores=n_cores
    )
    profile_fplay.rename(columns={
        'email': 'email_raw',
        'phone': 'phone_raw',
        'name': 'raw_name'
    }, inplace=True)

    # phone, email (valid)
    print(">>> Loading dictionaries")
    profile_phones = profile_fplay['phone_raw'].drop_duplicates().dropna()
    profile_emails = profile_fplay['email_raw'].drop_duplicates().dropna()
    profile_names = profile_fplay['raw_name'].drop_duplicates().dropna()

    # phone, email (valid)
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
    )

    # info
    print(">>> Processing Info")
    profile_fplay = profile_fplay[['uid', 'phone', 'email', 'name']]
    profile_fplay = profile_fplay.rename(columns={'uid': 'user_id'})
    profile_fplay = profile_fplay.sort_values(
        by=['user_id'], ascending=False)
    profile_fplay = profile_fplay.drop_duplicates(
        subset=['user_id'], keep='first')

    # merge get phone, email (valid)
    print(">>> Merging phone, email, name")
    profile_fplay = pd.merge(
        profile_fplay.set_index('phone_raw'),
        valid_phone.set_index('phone_raw'),
        left_index=True, right_index=True,
        how='left',
        sort=False
    ).reset_index(drop=False)

    profile_fplay = pd.merge(
        profile_fplay.set_index('email_raw'),
        valid_email.set_index('email_raw'),
        left_index=True, right_index=True,
        how='left',
        sort=False
    ).reset_index(drop=False)

    profile_fplay = pd.merge(
        profile_fplay.set_index('raw_name'),
        dict_name_lst.set_index('raw_name'),
        left_index=True, right_index=True,
        how='left',
        sort=False
    ).rename(columns={
        'enrich_name': 'name'
    }).reset_index(drop=False)

    # Refilling info
    cant_predict_name_mask = profile_fplay['name'].isna()
    profile_fplay.loc[
        cant_predict_name_mask,
        'name'
    ] = profile_fplay.loc[
        cant_predict_name_mask,
        'raw_name'
    ]
    profile_fplay['name'] = profile_fplay['name'].replace(dict_trash)

    # customer_type
    print(">>> Extracting customer type")
    profile_fplay = process_extract_name_type(
        profile_fplay,
        name_col='name',
        n_cores=n_cores,
        logging_info=False
    )

    # drop name is username_email
    print(">>> Extra Cleansing Name")
    profile_fplay = remove_same_username_email(
        profile_fplay,
        name_col='name',
        email_col='email'
    )

    # clean name, extract_pronoun
    condition_name = (profile_fplay['customer_type'] == 'customer')\
        & (profile_fplay['name'].notna())

    profile_fplay = extracting_pronoun_from_name(
        profile_fplay,
        condition=condition_name,
        name_col='name',
    )

    # is full name
    print(">>> Checking Full Name")
    profile_fplay.loc[profile_fplay['last_name'].notna(
    ) & profile_fplay['first_name'].notna(), 'is_full_name'] = True
    profile_fplay['is_full_name'] = profile_fplay['is_full_name'].fillna(False)
    profile_fplay = profile_fplay.drop(
        columns=['last_name', 'middle_name', 'first_name'])

    # add info
    print(">>> Adding Temp Info")
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
    profile_fplay = profile_fplay.rename(columns = {'user_id': 'uid'})

    # Map back customer type
    profile_fplay['customer_type'] =\
        profile_fplay['customer_type'].map({
            'customer': 'Ca nhan',
            'company': 'Cong ty',
            'medical': 'Benh vien - Phong kham',
            'edu': 'Giao duc',
            'biz': 'Ho kinh doanh'
        })
    # Fill 'Ca nhan'
    profile_fplay.loc[
        (profile_fplay['name'].notna())
        & (profile_fplay['customer_type'].isna()),
        'customer_type'
    ] = 'Ca nhan'

    # Save
    profile_fplay['d'] = date_str
    profile_fplay.to_parquet(
        f'{PREPROCESS_PATH}/{f_group}.parquet',
        filesystem=hdfs, index=False,
        partition_cols='d'
    )

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
    most_ip_fplay.to_parquet(UTILS_PATH + '/fplay_location_most.parquet', 
                             index=False, filesystem=hdfs)

if __name__ == '__main__':

    DAY = sys.argv[1]
    UnifyFplay(DAY, n_cores=10)
