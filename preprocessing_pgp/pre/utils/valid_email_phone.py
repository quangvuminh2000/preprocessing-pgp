
import pandas as pd
import numpy as np
from functools import partial
from itertools import repeat
from multiprocessing import Pool
from datetime import datetime, timedelta
from unidecode import unidecode
import multiprocessing as mp
import string

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

import sys
sys.path.append('/bigdata/fdp/cdp/cdp_pages/scripts_hdfs/pre/utils/email2info/')
import email_get_info

sys.path.append('/bigdata/fdp/cdp/cdp_pages/scripts_hdfs/pre/utils/')
import preprocess_lib

import sys
sys.path.append('/bigdata/fdp/cdp/cdp_pages/scripts_hdfs/pre/utils/fill_accent_name/scripts')
from enrich_name import process_enrich, fill_accent

ROOT_PATH = '/data/fpt/ftel/cads/dep_solution/sa/cdp/core'

# from vistickedword import split_words
# import wordninja

# PREPARE
subphone_vn = pd.read_parquet('/data/fpt/ftel/cads/dep_solution/sa/cdp/data/vn_sub_phone.parquet',
                              filesystem=hdfs).set_index('PhoneVendor')
subphone_vn = ('0' + subphone_vn.astype(str))
df_subtele_vn = pd.read_parquet('/data/fpt/ftel/cads/dep_solution/sa/cdp/data/vn_sub_telephone.parquet',
                                filesystem=hdfs)
SUB_PHONE_10NUM = sorted(subphone_vn['NewSubPhone'].unique())
SUB_PHONE_11NUM = [x for x in sorted(subphone_vn['OldSubPhone'].unique()) if len(x)==4]
SUB_TELEPHONE = sorted(df_subtele_vn['MaVung'].unique())
DICT_4SUBPHONE = subphone_vn[subphone_vn['OldSubPhone'].map(lambda x: len(x))==4].set_index('OldSubPhone').to_dict()['NewSubPhone']
EMAIL_CONFIGS = {
    'gmail': {
        'domains': ['gmail.com', 'gmail.com.vn'],
        'regex': '^[a-z0-9][a-z0-9\.]{4,28}[a-z0-9]@[a-z0-9]{2,}(?:\.[a-z0-9]{2,12}){1,2}$'
    },
    'yahoo': {
        'domains': ['yahoo.com', 'yahoo.com.vn'],
        'regex': '^[a-z][a-z0-9_\.]{2,30}[a-z0-9]@[a-z0-9]{2,}(?:\.[a-z0-9]{2,12}){1,2}$'
    },
    'ms': {
        'domains': ['hotmail.com', 'outlook.com', 'outlook.com.vn'],
        'regex': '^[a-z][a-z0-9-_\.]{2,62}[a-z0-9-_]@[a-z0-9]{2,}(?:\.[a-z0-9]{2,12}){1,2}$'
    },
    'fpt': {
        'domains': ['fpt.com.vn', 'fpt.edu.vn', 'hcm.fpt.vn', 'fpt.vn', 'fpt.net', 'fpt.aptech.ac.vn'],
        'regex': '^[a-z0-9][a-z0-9_\.]{2,31}[a-z0-9]@[a-z0-9]{2,}(?:\.[a-z0-9]{2,12}){1,2}$'
    }
}
COMMON_EMAIL_REGEX = '^[a-z0-9][a-z0-9_\.]{4,31}[a-z0-9]@[a-z0-9]{2,}(?:\.[a-z0-9]{2,12}){1,2}$'

# Only allow numbers and characters.
EDU_EMAIL_REGEX = '^[0-9a-z]+@[0-9a-z\.]'
CHECK_DOMAINS = []
for email_service in EMAIL_CONFIGS.values():
    CHECK_DOMAINS += email_service['domains']


# FUNCTION
def check_valid_phone(f_phones: pd.DataFrame, phone_col: str='phone') -> pd.DataFrame: 
    # Replace any characters that are not number with empty string.
    f_phones[phone_col] = f_phones[phone_col].str.replace(r'[^0-9]', '', regex=True).str.strip().str.replace('\s+', '', regex=True)
    f_phones['phone_length'] = f_phones[phone_col].map(lambda x: len(str(x))) 
    # Phone length validation: currently support phone number with length of 10 and 11.
    # Also, phone prefix has to be in the sub-phone dictionary.
    f_phones.loc[
        (
            (f_phones['phone_length']==10)
            & (f_phones[phone_col].str[:3].isin(SUB_PHONE_10NUM))
        ),
        'is_phone_valid'
    ] = True
    f_phones.loc[
        (
            (f_phones['phone_length']==11) 
            & (f_phones[phone_col].str[:4].isin(SUB_PHONE_11NUM))
        ), 
        'is_phone_valid'
    ] = True
    f_phones = f_phones.reset_index(drop=True)

    # Correct phone numbers with old phone number format.
    f_phones.loc[
        (f_phones['phone_length']==11) & (f_phones['is_phone_valid']==True), 
        'phone_convert'
    ] = f_phones.loc[
        (f_phones['phone_length']==11) & (f_phones['is_phone_valid']==True), 
        phone_col
    ].map(lambda x: DICT_4SUBPHONE[x[:4]] + x[4:] if (x[:4] in SUB_PHONE_11NUM) else None)

    # Check for tele-phone.
    f_phones.loc[
        (
            (f_phones['phone_length'] == 11) 
            & (f_phones[phone_col].str[:3].isin(SUB_TELEPHONE))
        ),
        'is_phone_valid'
    ] = True

    f_phones['is_phone_valid'] = f_phones['is_phone_valid'].fillna(False)
    f_phones = f_phones.drop('phone_length', axis=1)
    f_phones.loc[
        f_phones['is_phone_valid'] & f_phones['phone_convert'].isna(),
        'phone_convert'
    ] = f_phones[phone_col]

    return f_phones

def check_valid_email(f_emails: pd.DataFrame, email_col: str='email')-> pd.DataFrame:
    # Replace Nones with np.nan.
    repls = {
        'nan': np.nan,
        'none': np.nan,
        '': np.nan,
        None: np.nan
    }
    f_emails[email_col + '_convert'] = f_emails[email_col].str.lower().str.strip().str.replace('\s+', '', regex=True).replace(repls)

    email_col += '_convert'
    # Split email domain, group.
    email_splits = f_emails[email_col].str.split('@', n=1, expand=True)
    f_emails['email_name'] = email_splits[0]
    f_emails['email_group'] = email_splits[1]
    f_emails['email_len'] = f_emails['email_name'].map(lambda x: len(str(x)))
    # Emails must contain at least one character [a-z].
    contains_al_one_char_regex = '(?=.*[a-z])'
    # Emails can not contains accents like "á", "ả", etc.
    accented = (
        f_emails['email'].str.encode('ascii', errors='ignore') 
        != f_emails['email'].str.encode('ascii', errors='replace')
    )
    f_emails['is_email_valid'] = (
        f_emails['email_name'].str.match(contains_al_one_char_regex, na=False)
        & ~accented
    )

    for email_service in EMAIL_CONFIGS.values():
        # Apply different rule for different domain.
        f_emails.loc[
            (
                f_emails['is_email_valid']
                & f_emails['email_group'].isin(email_service['domains'])
            ),
            'is_email_valid'
        ] = f_emails[email_col].str.contains(email_service['regex'], regex=True, na=False)

    # Common email: not in large email companies.
    f_emails.loc[
        (
            f_emails['is_email_valid']
            & ~f_emails['email_group'].isin(CHECK_DOMAINS)
        ),
        'is_email_valid'
    ] = f_emails[email_col].str.contains(COMMON_EMAIL_REGEX, regex=True, na=False)

    # Student emails are allowed to start with number.
    f_emails.loc[
        f_emails['email_group'].str.contains('edu', regex=True, na=False)
        & f_emails[email_col].str.contains(EDU_EMAIL_REGEX, regex=True, na=False),
        'is_email_valid'
    ] = True

    # Edge cases: Apple and Sendo autoemail.
    f_emails.loc[
        f_emails[email_col].str.contains('@privaterelay.appleid.com|[0-9]+\_autoemail', regex=True, na=False),
        'is_email_valid'
    ] = False
    return f_emails

def _validate_phone_email(df_profile: pd.DataFrame, email_col: str='email', phone_col: str='phone') -> pd.DataFrame:
    df_profile = df_profile.pipe(check_valid_phone, phone_col=phone_col).pipe(check_valid_email, email_col=email_col)
    df_profile.drop(columns=['email_name', 'email_group', 'email_len'], inplace=True)
    df_profile.loc[
            (df_profile['is_phone_valid'] == True) 
            & (df_profile['phone_convert'].isna()),
            'phone_convert'
    ] = df_profile[phone_col]
    return df_profile

def validate_phone_email(df_profile: pd.DataFrame, n_cores: int, email_col: str='email', phone_col: str='phone') -> pd.DataFrame:
    if n_cores > 1:
        df_splits = np.array_split(df_profile, n_cores)
        pool = Pool(n_cores)
        try:
            #df_profile_val = pd.concat(pool.map(_validate_phone_email, df_splits))
            df_profile_val = pd.concat(pool.map(partial(_validate_phone_email, email_col=email_col, phone_col=phone_col), df_splits))
        except Exception as e:
            raise e
        finally:
            pool.close()
            pool.join()
        return df_profile_val
    return _validate_phone_email(df_profile)

def parallelize(data_frame, func, n_cores=16):
    df_split = np.array_split(data_frame, n_cores)
    pool = Pool(n_cores)
    try:
        data_frame = pd.concat(pool.map(func, df_split))
    except Exception as e:
        raise e
    finally:
        pool.close()
    return data_frame

def MetaDataPhone(valid_phone):    
    # LOAD DICTIONARY
    subphone_vn = pd.read_parquet('/data/fpt/ftel/cads/dep_solution/sa/cdp/data/vn_sub_phone.parquet',
                                  filesystem=hdfs, columns=['NewSubPhone', 'PhoneVendor'])
    subphone_vn.columns = ['sub_phone', 'phone_vendor']
    subphone_vn['sub_phone'] = '0' + subphone_vn['sub_phone'].astype(str)
    subphone_vn = subphone_vn.drop_duplicates()

    df_subtele_vn = pd.read_parquet('/data/fpt/ftel/cads/dep_solution/sa/cdp/data/vn_sub_telephone.parquet',
                                    filesystem=hdfs, columns=['MaVung', 'TinhThanh'])
    df_subtele_vn.columns = ['sub_phone', 'phone_vendor']
    df_subtele_vn['sub_phone'] = df_subtele_vn['sub_phone'].astype(str)
    df_subtele_vn = df_subtele_vn.drop_duplicates()
    
    # phone_type
    valid_phone.loc[valid_phone['phone'].str[:3] == '024', 'phone_type'] = 'landline'
    valid_phone.loc[valid_phone['phone'].str[:3].isin(df_subtele_vn['sub_phone'].unique()), 'phone_type'] = 'landline'
    valid_phone.loc[valid_phone['phone'].str[:3].isin(subphone_vn['sub_phone'].unique()) & 
                    valid_phone['phone_type'].isna(), 'phone_type'] = 'mobile phone'
    
    # add vender
    valid_phone_landline = valid_phone[valid_phone['phone_type'] == 'landline'].copy()
    valid_phone_mobile = valid_phone[valid_phone['phone_type'] == 'mobile phone'].copy()
    valid_phone_none = valid_phone[valid_phone['phone_type'].isna()].copy()


    valid_phone_landline['sub_phone'] = valid_phone_landline['phone'].str[:4].astype(str)
    valid_phone_landline = valid_phone_landline.merge(df_subtele_vn, how='left', on=['sub_phone'])
    valid_phone_landline.loc[(valid_phone_landline['phone'].str[:3] == '024') & 
                             valid_phone_landline['phone_vendor'].isna(), 'phone_vendor'] = 'Hà Nội'
    valid_phone_landline.loc[(valid_phone_landline['phone'].str[:3] == '028') & 
                             valid_phone_landline['phone_vendor'].isna(), 'phone_vendor'] = 'Thành phố Hồ Chí Minh'
 
    valid_phone_mobile['sub_phone'] = valid_phone_mobile['phone'].str[:3]
    valid_phone_mobile = valid_phone_mobile.merge(subphone_vn, how='left', on=['sub_phone'])

    valid_phone = pd.concat([valid_phone_landline, valid_phone_mobile, valid_phone_none], ignore_index=True)
    valid_phone = valid_phone.drop(columns=['sub_phone']).sample(frac=1)
    
    # beauty phone
    dict_beauty_phone = {
        'Ngũ Quý': ['00000', '11111', '22222', '33333', '44444', '55555', '66666', '77777', '88888', '99999'], 
        'Tứ Quý': ['0000', '1111', '2222', '3333', '4444', '5555', '6666', '7777', '8888', '9999'],
        'Tam Hoa': ['000', '111', '222', '333', '444', '555', '666', '777', '888', '999'], 
        'Số Tiến': ['0123', '1234', '2345', '3456', '4567', '5678', '6789'], 
        'Số Lùi': ['3210', '4321', '5432', '6543', '7654', '8765', '9876']
    }

    valid_phone['tail_phone_type'] = None
    for name_case, format_case in dict_beauty_phone.items():
        size_case = len(format_case[0])

        # assign
        valid_phone.loc[valid_phone['tail_phone_type'].isna() & 
                        valid_phone['phone'].str[-size_case:].isin(format_case), 'tail_phone_type'] = name_case

    valid_phone.loc[valid_phone['tail_phone_type'].isna() & valid_phone['phone'].notna(), 'tail_phone_type'] = 'Số Thường'

    # return
    return valid_phone

def FormatName(valid_email):
    # remove columns: customer_type, customer_type_detail
#     valid_email = valid_email.drop(columns=['customer_type', 'customer_type_detail'])
    
    # customer_type
    name_email = valid_email[valid_email['username'].notna()][['username']].copy().drop_duplicates()
    name_email.columns = ['name']
    name_email = preprocess_lib.ExtractCustomerType(name_email)
    
    # process format name
    com_name_email = name_email[name_email['customer_type'].notna() | 
                                (name_email['name'].str.split(' ').str.len() > 3)].copy()
    com_name_email['format_name'] = com_name_email['name']

    per_name_email = name_email[~name_email['name'].isin(com_name_email['name'].unique())].copy()[['name']]
    dict_trash = {'': None, 'Nan': None, 'nan': None, 'None': None, 'none': None, 'Null': None, 'null': None, "''": None}
    with mp.Pool(8) as pool:
        per_name_email[['last_name', 'middle_name', 'first_name']] = pool.map(preprocess_lib.SplitName, per_name_email['name'])
    columns = ['last_name', 'middle_name', 'first_name']
    per_name_email['format_name'] = per_name_email[columns].fillna('').agg(' '.join, axis=1).str.replace('(?<![a-zA-Z0-9]),', '', regex=True).str.replace('-(?![a-zA-Z0-9])', '', regex=True)
    per_name_email['format_name'] = per_name_email['format_name'].str.strip().replace(dict_trash)
    per_name_email = per_name_email.drop(columns=['last_name', 'middle_name', 'first_name'])

    name_email = pd.concat([com_name_email, per_name_email], ignore_index=True)
    name_email = name_email.rename(columns={'name': 'username'})
    
    # merge
    valid_email = valid_email.merge(name_email, how='left', on=['username'])
    valid_email['username'] = valid_email['format_name']
    valid_email.loc[valid_email['username'].isna(), 'username'] = None
    valid_email = valid_email.drop(columns=['format_name'])
    
    # fillna 'Ca nhan'
    valid_email.loc[valid_email['username'].notna() & 
                    valid_email['customer_type'].isna(), 'customer_type'] = 'Ca nhan'
    
    # Fill Accent + gender
    map_name = valid_email[valid_email['customer_type'] == 'Ca nhan'][['username']].drop_duplicates().copy()
    map_name = fill_accent(map_name, 'username')
    map_name = map_name[map_name['is_human'] == True][['username', 'predict_name']]
    
    map_name = preprocess_lib.Name2Gender(map_name, name_col='username')
    
    valid_email = valid_email.merge(map_name, how='left', on=['username'])
    
    valid_email.loc[valid_email['predict_name'].isna(), 'predict_name'] = valid_email['username']
    valid_email['username'] = valid_email['predict_name']
    
    valid_email.loc[valid_email['predict_gender'].isna(), 'predict_gender'] = valid_email['gender']
    valid_email['gender'] = valid_email['predict_gender']
    
    valid_email = valid_email.drop(columns=['predict_name', 'predict_gender'])
    
    # return
    return valid_email

def ValidPhoneEmail(date_str):
    raw_path = ROOT_PATH + '/raw'
    utils_path = ROOT_PATH + '/utils'
    FPTS = ['fo', 'fplay', 'ftel', 'fshop', 'longchau', 'sendo']
    today = date_str

    # VALID EMAIL
    emails_bank = pd.DataFrame()
    # Create emails bank from all companies.
    def load_email(cttv, key):
        emails = pd.read_parquet(f'{raw_path}/{cttv}.parquet/d={today}', 
                                filesystem=hdfs, 
                                columns=[key, 'email']
                               )
        emails.columns = ['id', 'email']
        emails['id'] = cttv + '-' + emails['id'].astype(str)
        return emails
    
    
    dict_cttv = {'ftel': 'contract_ftel',
                 'fo': 'vne_id_fo', 
                 'fplay': 'user_id_fplay',
                 'fshop': 'cardcode_fshop',
                 'longchau': 'cardcode_longchau',
                 'sendo': 'id_sendo'
                }
        
    for cttv, key in dict_cttv.items():
        emails_bank = emails_bank.append(load_email(cttv, key), ignore_index=True)
    
    latest_check_emails = pd.read_parquet(f'{utils_path}/valid_email_latest.parquet', filesystem=hdfs)
    emails_bank = emails_bank.loc[~emails_bank['email'].isin(latest_check_emails['email_raw'])] # Only check new emails
    
    if emails_bank.empty == False:
        # Regex syntax
        emails_bank['email'] = emails_bank['email'].str.strip().str.replace('\s+', '', regex=True)
        emails_bank.loc[emails_bank['email'] == '', 'email'] = np.nan
        
        # Not check email nan
        emails_bank = emails_bank.loc[emails_bank.email.notna()]

        emails_bank['count_id'] = emails_bank.groupby('email')['id'].transform('nunique')
        raw_f_emails = parallelize(emails_bank.copy(), check_valid_email, 10)
    #     raw_f_emails = check_valid_email(emails_bank.copy(), 'email')

        # Mapping email
        raw_mapping_emails = raw_f_emails[
            raw_f_emails['email'].notna()
        ][['email', 'email_convert']].copy().drop_duplicates(subset=['email'], keep='first')
        raw_mapping_emails.columns = ['email_raw', 'email']

        raw_f_emails = raw_f_emails.drop(columns=['email'])
        raw_f_emails = raw_f_emails.rename(columns={'email_convert': 'email'})

        # Pass syntax
        f_emails = raw_f_emails[
            raw_f_emails['is_email_valid']
        ][['id', 'email']].copy().drop_duplicates()

        raw_stats_email = f_emails.groupby(by=['email'])['id'].agg(count_id='count').reset_index()
        stats_email = raw_stats_email.copy()

        # Create 2 dict email
        email_ok_cdp = pd.DataFrame()
        email_warning_cdp = pd.DataFrame()

        # Use data account FO
        ## load data & preprocess
        profile_fo = pd.read_parquet('/data/fpt/fdp/fo/dwh/user_profile.parquet/', filesystem=hdfs, 
                                     columns=['vne_id', 'name', 'email', 'phone', 'age', 'gender', 
                                              'address', 'cookie', 'fo_id', 'ltv_total', 'user_login', 
                                              'created_date', 'last_visited', 'last_location'])
        email_fo = profile_fo[['name', 'email', 'user_login',]].drop_duplicates().copy()

        email_fo['user_login'] = email_fo['user_login'].str.strip()
        email_fo.loc[email_fo['user_login'] == '', 'user_login'] = None

        email_fo['name'] = email_fo['name'].str.strip()
        email_fo.loc[email_fo['name'] == '', 'name'] = None
        email_fo.loc[email_fo['name'].str.contains('^[0-9]*$', case=False, na=False), 'name'] = None

        email_fo['username_email'] = email_fo['email'].str.split('@').str[0]
        email_fo.loc[email_fo['name'].isna(), 'name'] = email_fo['username_email']

        ## level 1: name (legal), user_login (facebook, google)
        email_ok_fo = pd.DataFrame()
        email_ok_fo['email'] = email_fo[
            (email_fo['email'].isin(stats_email['email'])) | 
            (email_fo['name'] != email_fo['username_email']) | 
            (email_fo['user_login'].notna())
        ]['email'].tolist()
        email_ok_fo['description'] = 'Ok - FO (legal name or login)'

        email_ok_cdp = email_ok_cdp.append(email_ok_fo, ignore_index=True)
        stats_email = stats_email[~stats_email['email'].isin(email_ok_cdp['email'])]

        ## level 2: email auto => warning
        email_warning_fo = pd.DataFrame()
        temp_profile_fo = profile_fo[profile_fo['email'].isin(stats_email['email']) & 
                                     (profile_fo['name'].str.strip() == '') &
                                     (profile_fo['cookie'].str.strip() != '')].copy()

        ### vừa tạo + không dùng
        temp_profile_fo = temp_profile_fo[
            temp_profile_fo['created_date'].dt.date == temp_profile_fo['last_visited'].dt.date]
        stats_temp_profile_fo = temp_profile_fo.groupby(by=['cookie'])['vne_id'].agg(count_vne_id='count').reset_index()

        ### cùng lúc, 1 cookie nhiều account
        cookies_chaos = stats_temp_profile_fo[stats_temp_profile_fo['count_vne_id'] > 1]['cookie'].tolist()
        email_chaos_fo = temp_profile_fo[temp_profile_fo['cookie'].isin(cookies_chaos)]['email'].tolist()

        ### email chỉ xài ở 1 CTTV thì có khả năng là email rác nhiều hơn.
        email_warning_fo['email'] = stats_email[
            stats_email['email'].isin(email_chaos_fo) & 
            (stats_email['count_id'] == 1)
        ]['email'].tolist()
        email_warning_fo['description'] = 'Warning - FO (auto email)'

        ### save
        email_warning_cdp = email_warning_cdp.append(email_warning_fo, ignore_index=True)
        stats_email = stats_email[~stats_email['email'].isin(email_warning_cdp['email'])]

        # Use data account FPLAY
        profile_fplay = pd.read_parquet(f"{raw_path}/fplay.parquet/d={today}", filesystem=hdfs,
                                        columns=['user_id_fplay', 'phone', 'email', 'name'], 
                                        filters=[('email', 'in', stats_email['email'].unique())]).rename(columns={'user_id_fplay': 'id'})

        email_fplay = profile_fplay[profile_fplay['phone'].notna() & profile_fplay['email'].notna()].copy()
        email_fplay = email_fplay[['name', 'email']].drop_duplicates()

        email_fplay['name'] = email_fplay['name'].str.strip()
        email_fplay.loc[email_fplay['name'] == '', 'name'] = None
        email_fplay.loc[email_fplay['name'].str.contains('^[0-9]*$', case=False, na=False), 'name'] = None

        email_fplay['username_email'] = email_fplay['email'].str.split('@').str[0]
        email_fplay.loc[email_fplay['username_email'].str.contains('^[0-9]*$', case=False, na=False), 'username_email'] = None
        email_fplay = email_fplay[email_fplay['username_email'].notna()].drop_duplicates()
        email_fplay.loc[email_fplay['name'].isna(), 'name'] = email_fplay.loc[email_fplay['name'].isna(), 'username_email']

        email_fplay['domain_email'] = email_fplay['email'].str.split('@').str[1]

        ## level 1: name, phone, email
        email_ok_fplay = pd.DataFrame()
        email_ok_fplay['email'] = email_fplay[
            (email_fplay['name'] != email_fplay['username_email'])
        ]['email'].tolist()
        email_ok_fplay['description'] = 'Ok - FPLAY (phone, email & legal name)'
        email_ok_fplay = email_ok_fplay[email_ok_fplay['email'].isin(stats_email['email']) &
                                        ~email_ok_fplay['email'].isin(email_ok_cdp['email'])]

        email_ok_cdp = email_ok_cdp.append(email_ok_fplay, ignore_index=True)
        stats_email = stats_email[~stats_email['email'].isin(email_ok_cdp['email'])].copy()

        ## level 2: have phone
        email_ok_fplay = pd.DataFrame()

        ### nếu người dùng có sđt, điền email => khả năng cao hợp lệ
        email_ok_fplay['email'] = profile_fplay[
            profile_fplay['phone'].notna()
        ]['email'].tolist()

        email_ok_fplay['description'] = 'Ok - FPLAY (phone & email)'
        email_ok_fplay = email_ok_fplay[~email_ok_fplay['email'].isin(email_ok_cdp['email'])]

        email_ok_cdp = email_ok_cdp.append(email_ok_fplay, ignore_index=True)
        stats_email = stats_email[~stats_email['email'].isin(email_ok_cdp['email'])].copy()

        ## level 3: Have name 2 words and valid
        ### user chỉnh lại tên 
        word_name = pd.read_parquet('/data/fpt/ftel/cads/dep_solution/user/namdp11/name2gender/dataset/word_name.parquet',
                                    filesystem=hdfs)['word_name'].tolist()
        def CheckName(text):
            text = unidecode(text).lower()
            words_text = unidecode(text).split(' ')
            for word_text in words_text:
                if word_text in word_name:
                    return True
            return False

        temp_profile_fplay = profile_fplay[profile_fplay['email'].isin(stats_email['email'])].copy()
        temp_profile_fplay = temp_profile_fplay[temp_profile_fplay['name'].str.split(' ').str.len() > 1]
        temp_profile_fplay['name_valid'] = temp_profile_fplay['name'].apply(CheckName)

        email_ok_fplay = pd.DataFrame()
        email_ok_fplay['email'] = temp_profile_fplay[temp_profile_fplay['name_valid'] == True]['email'].tolist()
        email_ok_fplay['description'] = 'Ok - FPLAY (email & legal name)'
        email_ok_fplay = email_ok_fplay[~email_ok_fplay['email'].isin(email_ok_cdp['email'])]

        email_ok_cdp = email_ok_cdp.append(email_ok_fplay, ignore_index=True)
        stats_email = stats_email[~stats_email['email'].isin(email_ok_cdp['email'])].copy()

        # Domain email
        raw_stats_email['domain_email'] = raw_stats_email['email'].str.split('@').str[1]
        stats_domain_email = raw_stats_email.groupby(by=['domain_email'])['email'].agg(count_email='count').reset_index()
        bugs_domain_email = stats_domain_email[stats_domain_email['count_email'] == 1]['domain_email'].tolist()
        bugs_domain_email += ['gmal.com', 'gmial.com', 'gnail.com']

        stats_email['domain_email'] = stats_email['email'].str.split('@').str[1]

        email_warning_domain = pd.DataFrame()
        email_warning_domain['email'] = stats_email[stats_email['domain_email'].isin(bugs_domain_email)]['email'].tolist()
        email_warning_domain['description'] = 'Warning - Domain email'
        email_warning_domain = email_warning_domain[~email_warning_domain['email'].isin(email_warning_cdp['email'])]

        email_warning_cdp = email_warning_cdp.append(email_warning_domain, ignore_index=True)
        stats_email = stats_email[~stats_email['email'].isin(email_warning_cdp['email'])].copy()

        # Check dict email (extract name)
        email_dict = pd.read_parquet('/data/fpt/ftel/cads/dep_solution/sa/cdp/data/email_dict_valid_latest.parquet',
                                     filesystem=hdfs, filters=[('email', 'in', stats_email['email'].unique())])
        email_extract = email_dict[email_dict['username'].notna()]['email'].tolist()

        temp_ok_email_extract = pd.DataFrame()
        temp_ok_email_extract['email'] = email_extract
        temp_ok_email_extract['description'] = 'Ok - Email2Name'
        temp_ok_email_extract = temp_ok_email_extract[~temp_ok_email_extract['email'].isin(email_ok_cdp['email'])]

        email_ok_cdp = email_ok_cdp.append(temp_ok_email_extract, ignore_index=True)
        stats_email = stats_email[~stats_email['email'].isin(email_ok_cdp['email'])].copy()

        # Backup
        email_ok_backup = pd.DataFrame()
        email_ok_backup['email'] = list(stats_email['email'].unique())
        email_ok_backup['description'] = 'Ok - Backup'
        email_ok_backup = email_ok_backup[~email_ok_backup['email'].isin(email_ok_cdp['email'])]
        email_ok_cdp = email_ok_cdp.append(email_ok_backup, ignore_index=True)

        # Concat
        email_warning_cdp['use'] = False
        email_ok_cdp['use'] = True
        final_stats_email = email_warning_cdp.append(email_ok_cdp, ignore_index=True)
        final_stats_email = final_stats_email.drop_duplicates(subset=['email'], keep='first')
        final_stats_email = raw_stats_email[['email', 'count_id']].merge(final_stats_email, how='left', on=['email'])

        check_emails = final_stats_email[['email', 'description', 'use']].copy()
        check_emails.columns = ['email', 'description', 'is_email_valid']
        check_emails = raw_mapping_emails.merge(check_emails, how='left', on=['email'])
        check_emails.loc[check_emails['is_email_valid'].isna(), 'is_email_valid'] = False
        check_emails.loc[check_emails['description'].isna(), 'description'] = 'Warning - Syntax email'
        check_emails.loc[check_emails['is_email_valid'] == False, 'email'] = None
    else:
        check_emails = pd.DataFrame()

    # VALID PHONE
    phones_bank = pd.DataFrame()

    # Create phone bank from all companies.
    def load_phone(cttv, key):
        phones = pd.read_parquet(f'{raw_path}/{cttv}.parquet/d={today}', 
                                filesystem=hdfs, 
                                columns=[key, 'phone']
                               )
        phones.columns = ['id', 'phone']
        phones['id'] = cttv + '-' + phones['id'].astype(str)
        return phones
    
    dict_cttv = {'ftel': 'contract_ftel',
                 'fo': 'vne_id_fo', 
                 'fplay': 'user_id_fplay',
                 'fshop': 'cardcode_fshop',
                 'longchau': 'cardcode_longchau',
                 'sendo': 'id_sendo'
                }
        
    for cttv, key in dict_cttv.items():
        phones_bank = phones_bank.append(load_phone(cttv, key), ignore_index=True)

    phones_bank = phones_bank[phones_bank['phone'].notna()][['phone']].drop_duplicates().copy()
    
    latest_check_phones = pd.read_parquet(f'{utils_path}/valid_phone_latest.parquet', filesystem=hdfs)
    phones_bank = phones_bank[~phones_bank['phone'].isin(latest_check_phones['phone_raw'].unique())] # # Only check new phones
    
    if phones_bank.empty == False:
        check_phones = check_valid_phone(phones_bank, 'phone')
        check_phones.columns = ['phone_raw', 'is_phone_valid', 'phone']
    else:
        check_phones = pd.DataFrame()
    
    # UPDATE (lastest)
#     latest_check_emails = pd.read_parquet(f'{utils_path}/valid_email_latest.parquet', filesystem=hdfs)
#     latest_check_phones = pd.read_parquet(f'{utils_path}/valid_phone_latest.parquet', filesystem=hdfs)
    
#     new_check_emails = check_emails[~check_emails['email_raw'].isin(latest_check_emails['email_raw'].unique())]
#     new_check_phones = check_phones[~check_phones['phone_raw'].isin(latest_check_phones['phone_raw'].unique())]
    new_check_emails = check_emails.copy()
    new_check_phones = check_phones.copy()

    if new_check_emails.empty == False:
        email_input = new_check_emails[['email_raw']].rename(columns={'email_raw': 'email'}).copy()
        email_output = email_get_info.pipeline_email_to_info(email_input.copy())
        email_to_info = email_input.rename(columns={'email': 'email_raw'})
        email_to_info['email'] = email_to_info['email_raw'].str.lower()
        email_to_info = email_to_info.merge(email_output, how='left', on=['email'])
        email_to_info = email_to_info.drop(columns=['email'])
        
        new_check_emails = new_check_emails.merge(email_to_info, how='left', on=['email_raw'])
        new_check_emails = FormatName(new_check_emails)
        new_check_emails.loc[new_check_emails['customer_type'] != 'Ca nhan', 'gender'] = None
        new_check_emails['username_iscertain'] = new_check_emails['username_iscertain'].fillna(False)
        new_check_emails['is_autoemail'] = new_check_emails['is_autoemail'].fillna(False)
        new_check_emails['export_date'] = today
        
        latest_check_emails = pd.concat([latest_check_emails, new_check_emails], ignore_index=True)
        latest_check_emails.loc[latest_check_emails['is_email_valid'] == False, 'username_iscertain'] = False
#         latest_check_emails = FormatName(latest_check_emails)
        
    if new_check_phones.empty == False:
        new_check_phones = MetaDataPhone(new_check_phones)
        new_check_phones['export_date'] = today
        
        latest_check_phones = pd.concat([latest_check_phones, new_check_phones], ignore_index=True)
        
    # SAVE (lastest)
#     latest_check_emails.loc[latest_check_emails['customer_type'] != 'Ca nhan', 'gender'] = None
    latest_check_emails = latest_check_emails.drop_duplicates(subset=['email_raw'], keep='first')
    latest_check_emails.to_parquet(f'{utils_path}/valid_email_latest.parquet', filesystem=hdfs, index=False)
    
    latest_check_phones = latest_check_phones.drop_duplicates(subset=['phone_raw'], keep='first')
    latest_check_phones.to_parquet(f'{utils_path}/valid_phone_latest.parquet', filesystem=hdfs, index=False)
    
    # PRODUCT
    product_path = '/data/fpt/ftel/cads/dep_solution/sa/cdp/data'
    
    # emails
    pro_check_emails = latest_check_emails[(latest_check_emails['is_email_valid'] == True)]
    pro_check_emails = pro_check_emails.drop(columns=['email_raw', 'description', 'is_email_valid', 'email_name', 'export_date'])

    columns_email = [
        'email', 'username_iscertain', 'username', 'year_of_birth', 'phone', 'address', 
        'email_group', 'is_autoemail', 'gender', 'customer_type', # 'customer_type_detail'
    ]
    pro_check_emails = pro_check_emails[columns_email]

    pro_check_emails.to_parquet(f'{product_path}/valid_email_latest.parquet', filesystem=hdfs, index=False)

    # phones
    pro_check_phones = latest_check_phones[latest_check_phones['is_phone_valid'] == True]
    pro_check_phones = pro_check_phones.drop(columns=['is_phone_valid', 'export_date'])
    pro_check_phones.to_parquet(f'{product_path}/valid_phone_latest.parquet', filesystem=hdfs, index=False)

if __name__ == '__main__':
    date_str = sys.argv[1]
    ValidPhoneEmail(date_str)
#     MetaDataPhone(date_str)
