
import pandas as pd
import numpy as np
from unidecode import unidecode
from datetime import datetime, timedelta
import sys

import os
import subprocess
from pyarrow import fs

from preprocessing_pgp.name.type.extractor import process_extract_name_type

os.environ['HADOOP_CONF_DIR'] = "/etc/hadoop/conf/"
os.environ['JAVA_HOME'] = "/usr/jdk64/jdk1.8.0_112"
os.environ['HADOOP_HOME'] = "/usr/hdp/3.1.0.0-78/hadoop"
os.environ['ARROW_LIBHDFS_DIR'] = "/usr/hdp/3.1.0.0-78/usr/lib/"
os.environ['CLASSPATH'] = subprocess.check_output(
    "$HADOOP_HOME/bin/hadoop classpath --glob", shell=True).decode('utf-8')
hdfs = fs.HadoopFileSystem(
    host="hdfs://hdfs-cluster.datalake.bigdata.local", port=8020)

sys.path.append('/bigdata/fdp/cdp/cdp_pages/scripts_hdfs/pre')
from utils.preprocess_profile import (
    remove_same_username_email,
    cleansing_profile_name,
    extracting_pronoun_from_name
)
from utils.filter_profile import get_difference_data

ROOT_PATH = '/data/fpt/ftel/cads/dep_solution/sa/cdp/core'

# function get profile change/new


# def DifferenceProfile(now_df, yesterday_df):
#     difference_df = now_df[~now_df.apply(tuple, 1).isin(
#         yesterday_df.apply(tuple, 1))].copy()
#     return difference_df

# function unify profile


def UnifyFo(
    profile_fo: pd.DataFrame,
    n_cores: int = 1
):
    dict_trash = {'': None, 'Nan': None, 'nan': None, 'None': None,
                  'none': None, 'Null': None, 'null': None, "''": None}
    # * Cleansing
    print(">>> Cleansing profile")
    profile_fo = cleansing_profile_name(
        profile_fo,
        name_col='name',
        n_cores=n_cores
    )
    profile_fo.rename(columns={
        'email': 'email_raw',
        'phone': 'phone_raw',
        'name': 'raw_name'
    }, inplace=True)

    # * Loading dictionary
    print(">>> Loading dictionaries")
    profile_phones = profile_fo['phone_raw'].drop_duplicates().dropna()
    profile_emails = profile_fo['email_raw'].drop_duplicates().dropna()
    profile_names = profile_fo['raw_name'].drop_duplicates().dropna()

    # phone, email (valid)
    valid_phone = pd.read_parquet(
        f'{ROOT_PATH}/utils/valid_phone_latest_new.parquet',
        filters=[('phone_raw', 'in', profile_phones)],
        filesystem=hdfs,
        columns=['phone_raw', 'phone', 'is_phone_valid']
    )
    valid_email = pd.read_parquet(
        f'{ROOT_PATH}/utils/valid_email_latest_new.parquet',
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
            'gender',
            # 'customer_type'
        ]
    ).rename(columns={
        'gender': 'gender_enrich'
    })

    # info
    print(">>> Processing Info")
    profile_fo = profile_fo.rename(columns={'vne_id_fo': 'vne_id'})
    profile_fo = profile_fo.sort_values(
        by=['vne_id', 'last_active', 'active_date'], ascending=False)
    profile_fo = profile_fo.drop_duplicates(subset=['vne_id'], keep='first')
    profile_fo = profile_fo.drop(columns=['last_active', 'active_date'])

    # merge get phone, email (valid) and names
    print(">>> Merging phone, email, name")
    profile_fo = pd.merge(
        profile_fo.set_index('phone_raw'),
        valid_phone.set_index('phone_raw'),
        left_index=True, right_index=True,
        how='left',
        sort=False
    ).reset_index(drop=False)

    profile_fo = pd.merge(
        profile_fo.set_index('email_raw'),
        valid_email.set_index('email_raw'),
        left_index=True, right_index=True,
        how='left',
        sort=False
    ).reset_index(drop=False)

    profile_fo = pd.merge(
        profile_fo.set_index('raw_name'),
        dict_name_lst.set_index('raw_name'),
        left_index=True, right_index=True,
        how='left',
        sort=False
    ).rename(columns={
        'enrich_name': 'name'
    }).reset_index(drop=False)

    # Refilling info
    cant_predict_name_mask =\
        (profile_fo['name'].isna()) & (profile_fo['raw_name'].notna())
    profile_fo.loc[
        cant_predict_name_mask,
        'name'
    ] = profile_fo.loc[
        cant_predict_name_mask,
        'raw_name'
    ]
    profile_fo['name'] = profile_fo['name'].replace(dict_trash)

    # birthday
    print(">>> Processing Birthday")
    now_year = datetime.today().year
    profile_fo.loc[profile_fo['age'] < 0, 'age'] = np.nan
    # profile_fo.loc[profile_fo['age'] > profile_fo['age'].quantile(0.99), 'age'] = np.nan
    profile_fo.loc[profile_fo['age'] > 72, 'age'] = np.nan
    profile_fo.loc[profile_fo['age'].notna(), 'birthday'] =\
        (now_year - profile_fo[profile_fo['age'].notna()]['age'])\
        .astype(str).str.replace('.0', '', regex=False)
    profile_fo = profile_fo.drop(columns=['age'])
    profile_fo.loc[profile_fo['birthday'].isna(), 'birthday'] = None

    # gender
    print(">>> Processing Gender")
    profile_fo['gender'] = profile_fo['gender'].replace(
        {'Female': 'F', 'Male': 'M', 'Other': None})

    # customer type from raw
    print(">>> Extracting customer type")
    profile_fo = process_extract_name_type(
        profile_fo,
        name_col='name',
        n_cores=n_cores,
        logging_info=False
    )

    # drop name is username_email
    print(">>> Extra Cleansing Name")
    profile_fo = remove_same_username_email(
        profile_fo,
        name_col='name',
        email_col='email'
    )
    # profile_fo['username_email'] = profile_fo['email'].str.split('@').str[0]
    # profile_fo.loc[profile_fo['name'] ==
    #                profile_fo['username_email'], 'name'] = None
    # profile_fo = profile_fo.drop(columns=['username_email'])

    # clean name, extract pronoun

    condition_name = (profile_fo['customer_type'] == 'customer')\
        & (profile_fo['name'].notna())
    profile_fo = extracting_pronoun_from_name(
        profile_fo,
        condition_name,
        name_col='name',
    )

    # name_process = NameProcess()
    # profile_fo.loc[
    #     condition_name,
    #     ['clean_name', 'pronoun']
    # ] = profile_fo.loc[condition_name, 'name']\
    #     .apply(name_process.CleanName).tolist()

    # profile_fo.loc[
    #     profile_fo['customer_type'] == 'customer',
    #     'name'
    # ] = profile_fo['clean_name']
    # profile_fo = profile_fo.drop(columns=['clean_name'])

    # is full name
    print(">>> Checking Full Name")
    profile_fo.loc[profile_fo['last_name'].notna(
    ) & profile_fo['first_name'].notna(), 'is_full_name'] = True
    profile_fo['is_full_name'] = profile_fo['is_full_name'].fillna(False)
    profile_fo = profile_fo.drop(
        columns=['last_name', 'middle_name', 'first_name'])

    # valid gender by model
    print(">>> Validating Gender")
    profile_fo.loc[
        profile_fo['customer_type'] != 'customer',
        'gender'
    ] = None
    # profile_fo.loc[profile_fo['gender'].notna() & profile_fo['name'].isna(), 'gender'] = None
    profile_fo.loc[
        (profile_fo['gender'].notna())
        & (profile_fo['gender'] != profile_fo['gender_enrich']),
        'gender'
    ] = None

    # address, city
    print(">>> Processing Address")
    norm_fo_city = pd.read_parquet('/data/fpt/ftel/cads/dep_solution/user/namdp11/scross_fill/runner/refactor/material/ftel_provinces.parquet',
                                   filesystem=hdfs)
    norm_fo_city.columns = ['city', 'norm_city']
    profile_fo.loc[profile_fo['address'] == 'Not set', 'address'] = None
    profile_fo.loc[profile_fo['address'].notna(
    ), 'city'] = profile_fo.loc[profile_fo['address'].notna(), 'address'].apply(unidecode)
    profile_fo['city'] = profile_fo['city'].replace({
        'Ba Ria - Vung Tau': 'Vung Tau',
        'Thua Thien Hue': 'Hue',
        'Bac Kan': 'Bac Can',
        'Dak Nong': 'Dac Nong'
    })
    profile_fo = pd.merge(
        profile_fo.set_index('city'),
        norm_fo_city.set_index('city'),
        left_index=True, right_index=True,
        how='left',
        sort=False
    ).reset_index()
    profile_fo['city'] = profile_fo['norm_city']
    profile_fo = profile_fo.drop(columns=['norm_city'])
    profile_fo.loc[profile_fo['city'].isna(), 'city'] = None
    profile_fo['address'] = None

    # add info
    print(">>> Adding Temp Info")
    profile_fo['unit_address'] = None
    profile_fo['ward'] = None
    profile_fo['district'] = None
    columns = ['vne_id', 'phone_raw', 'phone', 'is_phone_valid',
               'email_raw', 'email', 'is_email_valid',
               'name', 'pronoun', 'is_full_name', 'gender',
               'birthday', 'customer_type',  # 'customer_type_detail',
               'address', 'unit_address', 'ward', 'district', 'city']
    profile_fo = profile_fo[columns]
    profile_fo = profile_fo.rename(columns={'vne_id': 'vne_id_fo'})
    # Fill 'Ca nhan'
    profile_fo['customer_type'] =\
    profile_fo['customer_type'].map({
        'customer': 'Ca nhan',
        'company': 'Cong ty',
        'medical': 'Benh vien - Phong kham',
        'edu': 'Giao duc',
        'biz': 'Ho kinh doanh'
    })
    profile_fo.loc[
        (profile_fo['name'].notna())
        & (profile_fo['customer_type'].isna()),
        'customer_type'
    ] = 'Ca nhan'

    # return
    return profile_fo

# function update profile (unify)


def UpdateUnifyFo(
    now_str:str,
    n_cores:int = 1
):
    # VARIABLES
    raw_path = ROOT_PATH + '/raw'
    unify_path = ROOT_PATH + '/pre'
    f_group = 'fo'
    yesterday_str = (datetime.strptime(now_str, '%Y-%m-%d') -
                     timedelta(days=1)).strftime('%Y-%m-%d')

    # load profile (yesterday, now)
    print(">>> Loading today and yesterday profile")
    info_columns = ['vne_id_fo', 'phone', 'email', 'name',
                    'gender', 'age', 'address', 'last_active', 'active_date']
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
    difference_profile = get_difference_data(now_profile, yesterday_profile)
    print(f"Number of new profile {difference_profile.shape}")

    # update profile
    profile_unify = pd.read_parquet(
        f'{unify_path}/{f_group}.parquet/d={yesterday_str}',
        filesystem=hdfs
    )
    if not difference_profile.empty:
        # get profile unify (old + new)
        new_profile_unify = UnifyFo(difference_profile, n_cores=n_cores)

        # synthetic profile
        profile_unify = pd.concat(
            [new_profile_unify, profile_unify],
            ignore_index=True
        )

    # arrange columns
    print(">>> Re-Arranging Columns")
    columns = [
        'vne_id_fo', 'phone_raw', 'phone', 'is_phone_valid',
        'email_raw', 'email', 'is_email_valid',
        'name', 'pronoun', 'is_full_name', 'gender',
        'birthday', 'customer_type',  # 'customer_type_detail',
        'address', 'unit_address', 'ward', 'district', 'city'
    ]
    profile_unify = profile_unify[columns]
    profile_unify['is_phone_valid'] =\
        profile_unify['is_phone_valid'].fillna(False)
    profile_unify['is_email_valid'] =\
        profile_unify['is_email_valid'].fillna(False)
    profile_unify = profile_unify.drop_duplicates(
        subset=['vne_id_fo', 'phone_raw', 'email_raw'],
        keep='first'
    )

    # save
    profile_unify['d'] = now_str
    profile_unify.to_parquet(
        f'{unify_path}/{f_group}_new.parquet',
        filesystem=hdfs, index=False,
        partition_cols='d'
    )

# function update ip (most)


def UnifyLocationIpFo():
    # MOST LOCATION IP
    dict_ip_path = '/data/fpt/ftel/cads/dep_solution/user/namdp11/scross_fill/runner/ip/dictionary'
    log_ip_path = '/data/fpt/ftel/cads/dep_solution/user/namdp11/scross_fill/runner/ip/fo'

    ip_location1 = pd.read_parquet(
        f'{dict_ip_path}/ip_location_batch_1.parquet', filesystem=hdfs)
    ip_location2 = pd.read_parquet(
        f'{dict_ip_path}/ip_location_batch_2.parquet', filesystem=hdfs)
    ip_location = pd.concat([ip_location1, ip_location2], ignore_index=True)
    # ip_location = ip_location1.append(ip_location2, ignore_index=True)
    ip_location = ip_location[['ip', 'name_province', 'name_district']]

    # update ip
    def IpFo(date):
        date_str = date.strftime('%Y-%m-%d')
        try:
            # load log ip
            log_df = pd.read_parquet(f"/data/fpt/fdp/fo/dwh/stag_access_features.parquet/d={date_str}",
                                     filesystem=hdfs, columns=['user_id', 'ip', 'isp']).drop_duplicates()
            log_df['date'] = date_str
            log_df.to_parquet(
                f'{log_ip_path}/ip_{date_str}.parquet', index=False, filesystem=hdfs)

            # add location
            location_df =\
                pd.merge(
                    log_df.set_index('ip'),
                    ip_location.set_index('ip'),
                    how='left',
                    left_index=True,
                    right_index=True
                ).reset_index()
            # log_df.merge(ip_location, how='left', on='ip')
            location_df.to_parquet(
                f'{log_ip_path}/location/ip_{date_str}.parquet', index=False, filesystem=hdfs)
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
    logs_ip_path = sorted([f.path for f in hdfs.get_file_info(
        fs.FileSelector(f'{log_ip_path}/location/'))])[-180:]
    ip_fo = pd.read_parquet(logs_ip_path, filesystem=hdfs)
    stats_ip_fo = ip_fo.groupby(by=['user_id', 'name_province', 'name_district'])[
        'date'].agg(num_date='count').reset_index()
    stats_ip_fo = stats_ip_fo.sort_values(
        by=['user_id', 'num_date'], ascending=False)
    most_ip_fo = stats_ip_fo.drop_duplicates(subset=['user_id'], keep='first')
    most_ip_fo.to_parquet(
        f'{ROOT_PATH}/utils/fo_location_most.parquet',
        index=False, filesystem=hdfs
    )


if __name__ == '__main__':

    now_str = sys.argv[1]
    UpdateUnifyFo(now_str, n_cores=5)
    UnifyLocationIpFo()
