
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from config.cfg import Config
import unicodedata
from string import punctuation
import pyarrow.parquet as pq
from pyarrow import fs
import subprocess
import os
import pickle
import string
import re
import numpy as np
import pandas as pd
import multiprocessing as mp
from unidecode import unidecode
from multiprocessing import Pool
from difflib import SequenceMatcher
from datetime import datetime, timedelta
from preprocess import clean_name_cdp
from enrich_name import process_enrich, fill_accent

from preprocessing_pgp.name.enrich_name import process_enrich
from preprocessing_pgp.name.gender.predict_gender import process_predict_gender

import sys
sys.path.append(
    '/bigdata/fdp/cdp/cdp_pages/scripts_hdfs/pre/utils/fill_accent_name/scripts')


os.environ['HADOOP_CONF_DIR'] = "/etc/hadoop/conf/"
os.environ['JAVA_HOME'] = "/usr/jdk64/jdk1.8.0_112"
os.environ['HADOOP_HOME'] = "/usr/hdp/3.1.0.0-78/hadoop"
os.environ['ARROW_LIBHDFS_DIR'] = "/usr/hdp/3.1.0.0-78/usr/lib/"
os.environ['CLASSPATH'] = subprocess.check_output(
    "$HADOOP_HOME/bin/hadoop classpath --glob", shell=True).decode('utf-8')
hdfs = fs.HadoopFileSystem(
    host="hdfs://hdfs-cluster.datalake.bigdata.local", port=8020)

os.environ['HTTP_PROXY'] = "http://proxy.hcm.fpt.vn:80/"
os.environ['HTTPS_PROXY'] = "http://proxy.hcm.fpt.vn:80/"

ROOT_PATH = '/data/fpt/ftel/cads/dep_solution/sa/cdp/core'


def BuildWordName():
    # load database name
    ext_data_name1 = pd.read_parquet(
        '/data/fpt/ftel/cads/dep_solution/user/namdp11/name2gender/dataset/ext_data.parquet', filesystem=hdfs)
    ext_data_name2 = pd.read_parquet(
        '/data/fpt/ftel/cads/dep_solution/user/namdp11/name2gender/dataset/ext_data_uit.parquet', filesystem=hdfs)

    ext_data_name1.columns = ['full_name', 'gender']
    ext_data_name1['full_name'] = ext_data_name1['full_name'].str.replace(
        '\s+', ' ', regex=True).str.title()
    ext_data_name1['first_name'] = ext_data_name1['full_name'].str.split(
        ' ').str[-1]
    ext_data_name1['last_name_group'] = ext_data_name1['full_name'].str.split(
        ' ').str[0]
    ext_data_name1['last_name'] = ext_data_name1['full_name'].str.split(
        ' ').str[:-1].str.join(' ')

    ext_data_name2['full_name'] = ext_data_name2['full_name'].str.replace(
        '\s+', ' ', regex=True).str.title()
    ext_data_name2 = ext_data_name2[[
        'full_name', 'gender', 'first_name', 'last_name_group', 'last_name']].copy()

    # stats freq
    ext_data_name1['full_name_unicecode'] = ext_data_name1['full_name'].str.lower(
    ).apply(unidecode)
    stats_word_name1 = ext_data_name1['full_name_unicecode'].str.split(
        expand=True).stack().value_counts().reset_index()
    stats_word_name1.columns = ['Word', 'Frequency']

    ext_data_name2['full_name_unicecode'] = ext_data_name2['full_name'].str.lower(
    ).apply(unidecode)
    stats_word_name2 = ext_data_name2['full_name_unicecode'].str.split(
        expand=True).stack().value_counts().reset_index()
    stats_word_name2.columns = ['Word', 'Frequency']

    # wordName
    stats_word_name = stats_word_name1.append(
        stats_word_name2, ignore_index=False)
    stats_word_name = stats_word_name.groupby(
        by=['Word'])['Frequency'].sum().reset_index()
    stats_word_name = stats_word_name[~((stats_word_name['Frequency'] < 5) |
                                        stats_word_name['Word'].str.contains('[^a-z]'))]
    word_name = set(stats_word_name['Word'])

    # return
    return word_name


def BuildLastName():
    # load stats lastname
    stats_lastname_vn = pd.read_parquet(
        '/data/fpt/ftel/cads/dep_solution/user/namdp11/name2gender/dataset/stats_lastname_vn.parquet', filesystem=hdfs)

    last_name_list1 = list(
        stats_lastname_vn[stats_lastname_vn['No'] <= 50]['Last_Name'].unique())
    last_name_list2 = ['Nguyễn', 'Trần', 'Lê', 'Phạm', 'Hoàng', 'Huỳnh', 'Phan',
                       'Vũ', 'Võ', 'Đặng', 'Bùi', 'Đỗ', 'Hồ', 'Ngô', 'Dương', 'Lý',
                       'Trương', 'Bùi', 'Đinh', 'Lương', 'Tạ', 'Quách', 'Hứa']
    last_name_list3 = last_name_list1
    last_name_list = list(set(last_name_list3) | set(
        last_name_list1) | set(last_name_list2))

    # return
    return last_name_list1, last_name_list2, last_name_list3, last_name_list


# Run build database Name
word_name = BuildWordName()
last_name_list1, last_name_list2, last_name_list3, last_name_list = BuildLastName()


def CountNameVN(text, word_name=word_name):
    try:
        text = unidecode(text.lower())

        # Check in word_name
        word_text = set(text.split())
        valid_words = list(word_text.intersection(word_name))

        count = 0
        for word in text.split():
            if word in valid_words:
                count += 1

        return count

    except:
        return -1

# CLEAN_NAME


def CleanName(raw_name, email):
    try:
        process_name = raw_name.lower().strip()
        # is email?
        regex_email = r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+'
        process_name = re.sub(regex_email, '', process_name)
        try:
            process_name = re.sub(email.split('@')[0], '', process_name)
        except:
            process_name = process_name
        # is phone?
        regex_phone = r'[0-9]*'
        process_name = re.sub(regex_phone, '', process_name)
        # special char
        process_name = re.sub(r'\(.*\)', ' ', process_name)
        process_name = re.sub(r'\.|,|-|_', ' ', process_name)
        process_name = re.sub(r'\s+', ' ', process_name)
        process_name = process_name.split(',')[0]
        process_name = process_name.split('-')[0]
        process_name = process_name.strip('-| |.|,|(|)')
        # fix typing
        regex_typing1 = '0[n|a|i|m|c|t][^wbmc\d ]'
        regex_typing2 = '[h|a|b|v|g|e|x|c]0[^0-9.-]'
        process_name = re.sub(regex_typing1, 'o', process_name)
        process_name = re.sub(regex_typing2, 'o', process_name)
        # pronoun
        regex_pronoun1 = r'^(?:\bkh\b|\bkhach hang\b|\bchị\b|\bchi\b|\banh\b|\ba\b|\bchij\b|\bc\b|\be\b|\bem\b|\bcô\b|\bco\b|\bchú\b|\bbác\b|\bbac\b|\bme\b|\bdì\b|\bông\b|\bong\b|\bbà\b|\ba\.|\bc\.)\s+'
        regex_pronoun2 = r'^(?:\bnội\b|\bngoại\b)\s+'
        regex_pronoun3 = r'^(?:\bvo anh\b|\bvo a\b|\bvo chu\b|\bbo anh\b|\bme anh\b|\bem anh\b|\bbo a\b|\bban chi\b|\bbo chi\b|\bban\b|\bck\b|\bvk\b)\s+'
        try:
            pronouns = re.findall(regex_pronoun1, process_name)
            pronoun = pronouns[0].strip()
        except:
            pronoun = None
        process_name = re.sub(regex_pronoun1, '', process_name)
        process_name = re.sub(regex_pronoun2, '', process_name)
        process_name = re.sub(regex_pronoun3, '', process_name)
        process_name = process_name.strip('-| |.|,|(|)')
        # defaut
        regex_default1 = r'người mua|người nhận|số ngoài danh bạ|nhập số điện thoại|giao hàng|test|[~!#$%^?]'
        regex_default2 = r'dung chung|ky thuat|du phong|dong tien|chu cu|chung phong|co quan|thu nhat|thu hai|so cong ty|chu moi|nhan su|dong nghiep|lien quan|em cua|may ban|so may|nghi lam|quan ly|dat so|su dung|nhan vien|chu nha|moi mua|dien thoai|chuyen di|lap dat|cung phong|nham so|hop dong|tong dai|can ho|ke toan|k co|so phu|lien he|lien lac|don di|so cu|so moi|ve que|khong dung|ben canh|ko co'
        regex_default3 = r'kh |khach hang|chu nha|cong ty|cc|nsd|vo kh|chong kh|so chu hd|so moi|nguoi moi tiep nhan'
        process_name = re.sub(regex_default1, '', process_name)
        process_name = re.sub(regex_default2, '', process_name)
        process_name = re.sub(regex_default3, '', process_name)
        process_name = process_name.strip('-| |.|,|(|)')
        process_name = re.sub(r'\s+', ' ', process_name)
        # is dict name VN
        pct_vn = CountNameVN(process_name)/len(process_name.split(' '))
        process_name = None if ((pct_vn < 0.8) |
                                (len(process_name) == 1) |
                                (len(process_name.split(' ')) > 6)
                                ) else process_name
        # add process (QuangVM9)
#         process_name = clean_name_cdp(process_name)
        # title
        process_name = process_name.title()
        return process_name, pronoun
    except:
        return None, None

# SPLIT_NAME


def SplitName(full_name):
    try:
        # Variable
        use_only_last_name = True
        full_name = full_name.replace('\s+', ' ').strip().title()
        last_name = ''
        middle_name = None
        first_name = None

        # Case 0: B
        if (full_name.split(' ') == 1):
            first_name = full_name
            return last_name, middle_name, first_name

        # Case 1: Nguyen Van C
        check_end_case1 = False
        while (check_end_case1 == False):
            for key_vi in last_name_list1:
                key_vi = key_vi + ' '

                is_case11 = (full_name.find(key_vi) == 0)
                is_case12 = (full_name.find(unidecode(key_vi)) == 0)
                is_case1 = is_case11 or is_case12
                if is_case1:
                    key = key_vi if is_case11 else unidecode(key_vi)

                    last_name = (last_name + ' ' + key).strip()
                    full_name = full_name.replace(key, '', 1).strip()

                    if (use_only_last_name == True):
                        check_end_case1 = True
                    break

                if (full_name.split(' ') == 1) or (key_vi.strip() == last_name_list1[-1]):
                    check_end_case1 = True

        # Case 2: Van D Nguyen
        if (last_name.strip() == ''):
            check_end_case2 = False
            while (check_end_case2 == False):
                for key_vi in last_name_list2:
                    key_vi = ' ' + key_vi

                    is_case21 = (len(full_name)-full_name.rfind(key_vi)
                                 == len(key_vi)) & (full_name.rfind(key_vi) != -1)
                    is_case22 = (len(full_name)-full_name.rfind(unidecode(key_vi)) == len(
                        unidecode(key_vi))) & (full_name.rfind(unidecode(key_vi)) != -1)

                    is_case2 = is_case21 or is_case22
                    if is_case2:
                        key = key_vi if is_case21 else unidecode(key_vi)

                        last_name = (key + ' ' + last_name).strip()
                        full_name = ''.join(full_name.rsplit(key, 1)).strip()

                        if (use_only_last_name == True):
                            check_end_case2 = True
                        break

                    if (full_name.split(' ') == 1) or (key_vi.strip() == last_name_list2[-1]):
                        check_end_case2 = True

        # Case 3: E Nguyen Van
        if (last_name.strip() == ''):
            temp_full_name = full_name
            temp_first_name = temp_full_name.split(' ')[0]
            temp_full_name = ' '.join(temp_full_name.split(' ')[1:]).strip()

            check_end_case3 = False
            while (check_end_case3 == False):
                for key_vi in last_name_list3:
                    key_vi = key_vi + ' '

                    is_case31 = (temp_full_name.find(key_vi) == 0)
                    is_case32 = (temp_full_name.find(unidecode(key_vi)) == 0)
                    is_case3 = is_case31 or is_case32
                    if is_case3:
                        key = key_vi if is_case31 else unidecode(key_vi)

                        last_name = (last_name + ' ' + key).strip()
                        temp_full_name = temp_full_name.replace(
                            key, '', 1).strip()

                        if (use_only_last_name == True):
                            check_end_case3 = True
                        break

                    if (full_name.split(' ') == 1) or (key_vi.strip() == last_name_list3[-1]):
                        check_end_case3 = True

            if (last_name.strip() != ''):
                first_name = temp_first_name
                middle_name = temp_full_name

                return last_name, middle_name, first_name

        # Fillna
        first_name = full_name.split(' ')[-1]
        try:
            full_name = ''.join(full_name.rsplit(first_name, 1)).strip()
            middle_name = full_name
        except:
            middle_name = None

        # Replace '' to None
        last_name = None if (last_name == '') else last_name
        middle_name = None if (middle_name == '') else middle_name
        first_name = None if (first_name == '') else first_name

        return last_name, middle_name, first_name

    except:
        return None, None, None


# NAME_2_CUSTOMER_TYPE
sys.path.insert(1, '/bigdata/fdp/cdp/script/')

config = Config()
config.load_cfg('/bigdata/fdp/cdp/script/config/customer_type.ini', 'ct')


def RunExtractCustomerType(profile,
                           type_name,
                           lv1_kws,
                           lv2_kws,
                           lv2_kws_map=None,
                           name_col='name',
                           exclude_regex=None
                           ):

    profile_opt = profile.copy()

    conditions = profile_opt['{}_en'.format(name_col)].str.contains(
        '|'.join([unidecode(kw) for kw in lv1_kws]), regex=True, na=False, case=False)
    if exclude_regex:
        conditions = conditions & ~profile_opt['{}_en'.format(name_col)].str.contains(
            exclude_regex, regex=True, na=False, case=False)

    profile = profile[~conditions]
    profile_opt = profile_opt[conditions]
    profile_opt['{}_en_clean'.format(name_col)] = profile_opt['{}_en'.format(
        name_col)].str.replace(rf'[{punctuation}]', '', regex=True)

    def kw_in_lv2(name):
        if lv2_kws_map:
            return [
                lv2_kws_map[lv2.lower()]
                for lv2 in lv2_kws
                if lv2.lower() in name.lower()
            ]

        return [
            lv2
            for lv2 in lv2_kws
            if lv2.lower() in name.lower()
        ]
    profile_opt['customer_type'] = type_name
#     profile_opt['customer_type_detail'] = profile_opt['{}_en_clean'.format(name_col)].apply(kw_in_lv2).apply(', '.join).replace('', np.nan)

    profile_after = pd.concat([
        profile,
        profile_opt
    ], ignore_index=True)

    return profile_after


def ExtractCustomerType(profile):
    name_col = 'name'

    profile['{}_en'.format(name_col)] = profile[name_col].apply(lambda name: unidecode(
        unicodedata.normalize('NFKD', ' '.join(name.split()))).lower() if isinstance(name, str) else name)
    profile['{}_len'.format(name_col)] = profile['{}_en'.format(name_col)].apply(
        lambda name: len(name.split(' ')) if isinstance(name, str) else 0)

    EXCLUDE_REGEXES = {
        'company': 'benh vien|ngan hang',
        'biz': None,
        'edu': None,
        'medical': None
    }

    CTYPE_NAMES = {
        'company': 'Cong ty',
        'biz': 'Ho kinh doanh',
        'edu': 'Giao duc',
        'medical': 'Benh vien - Phong kham'
    }

    profile_opt = (
        profile
        .assign(customer_type=None)
        .copy()
    )

    for ctype in ['company', 'biz', 'edu', 'medical']:

        lv1_kws = config.get('ct', ctype, 'lv1')
        lv2_kws = config.get('ct', ctype, 'lv2')
        lv2_kws_map = config.get('ct', ctype, 'lv2_map')

        # Exclude extracted profiles.
        exclude_cond = profile_opt['customer_type'].notna()
        if ctype == 'edu':
            exclude_cond = exclude_cond | (profile_opt['name_len'] < 4)

        profile_exclude = profile_opt[exclude_cond]
        profile_opt = profile_opt[~exclude_cond]

        profile_opt = pd.concat([
            RunExtractCustomerType(
                profile_opt, CTYPE_NAMES[ctype], lv1_kws, lv2_kws, lv2_kws_map, exclude_regex=EXCLUDE_REGEXES[ctype]),
            profile_exclude,
        ], ignore_index=True)

    profile_opt_final = (
        profile_opt
        .loc[:, ['name', 'customer_type']]  # , 'customer_type_detail']]
    )

    return profile_opt_final


def Name2Gender(df, name_col='name'):
    # copy
    process_df = df.copy()

    # process name
    process_df['processed_name'] = process_df[name_col].str.lower().str.replace(
        r'\d', ' ').str.replace(
        rf'[{string.punctuation}]', ' ').str.replace(
        r'\s+', ' ').str.strip()

    # check have_accented
    have_accented = (process_df['processed_name'].apply(
        unidecode) != process_df['processed_name'])

    # load model
    accented_model = pickle.loads(hdfs.open_input_file(
        f'/data/fpt/ftel/cads/dep_solution/user/namdp11/name2gender/accented/logistic_pipeline.pkl').read())
    non_accented_model = pickle.loads(hdfs.open_input_file(
        f'/data/fpt/ftel/cads/dep_solution/user/namdp11/name2gender/not_accented/logistic_pipeline.pkl').read())

    # predict gender (accented)
    accented_process_df = process_df[have_accented].copy()
    if (accented_process_df.empty == False):
        accented_predict_gender = accented_model.predict(
            accented_process_df['processed_name'].values)
        process_df.loc[accented_process_df.index,
                       'predict_gender'] = accented_predict_gender

    # predict gender (non accented)
    non_accented_process_df = process_df[~have_accented].copy()
    if (non_accented_process_df.empty == False):
        non_accented_predict_gender = non_accented_model.predict(
            non_accented_process_df['processed_name'].values)
        process_df.loc[non_accented_process_df.index,
                       'predict_gender'] = non_accented_predict_gender

    # replace
    process_df['predict_gender'] = process_df['predict_gender'].map({
                                                                    1: 'M', 0: 'F'})
    process_df = process_df.drop(columns=['processed_name'])

    # return
    return process_df

# FILL ACCENT


def UpdateDictName(date_str):
    # LOAD DATA
    # dict name
    dict_name_cdp = pd.read_parquet(
        ROOT_PATH + '/utils/dict_name_latest.parquet', filesystem=hdfs)
    # dict_name_cdp['unidecode_full_name'] = dict_name_cdp['full_name'].apply(
    #     unidecode)

    # Load data CDP
    pre_path = ROOT_PATH + '/pre'
    cttv_s = ['fo', 'fplay', 'ftel', 'sendo', 'fshop', 'longchau']
    raw_names = pd.DataFrame()

    # loop
    for name_cttv in cttv_s:
        # read data
        data_cttv = pd.read_parquet(f'{pre_path}/{name_cttv}.parquet/d={date_str}', filesystem=hdfs,
                                    columns=['name', 'gender'])

        # filters
        data_cttv = data_cttv[data_cttv['name'].notna()][[
            'name']].drop_duplicates()

        # append
        raw_names = pd.concat([raw_names, data_cttv], ignore_index=True)

    #! Load data Email -- MAYBE DEPRECATED IN FUTURE
    name_email = pd.read_parquet('/data/fpt/ftel/cads/dep_solution/sa/cdp/data/email_dict_latest.parquet',
                                 filesystem=hdfs, filters=[('username_iscertain', '=', True)])[['username']]
    name_email = name_email[name_email['username'].notna()].drop_duplicates()
    name_email.columns = ['name']
    name_email = name_email[~name_email['name'].isin(raw_names['name'])]

    # Concat raw data
    raw_names = pd.concat([raw_names, name_email], ignore_index=True)
    raw_names = raw_names.drop_duplicates()
    raw_names = raw_names[~raw_names['name'].isin(dict_name_cdp['full_name'])]

    # * Change to enrich name in package
    pre_names = process_enrich(
        raw_names,
        name_col='name',
        n_cores=1
    )
    pre_names.rename(columns={
        'final': 'enrich_name'
    })

    # ? CHANGE BLOCK
    # # Filters: name accent and not accent
    # raw_names['unidecode_full_name'] = raw_names['name'].apply(unidecode)
    # accent_raw_names = raw_names[raw_names['name'] !=
    #                              raw_names['unidecode_full_name']][['name']].copy()
    # not_accent_raw_names = raw_names[raw_names['name'] ==
    #                                  raw_names['unidecode_full_name']][['name']].copy()

    # # NAME ACCENT
    # # Add source
    # accent_raw_names = accent_raw_names[~accent_raw_names['name'].isin(
    #     dict_name_cdp['full_name'])].copy()
    # accent_raw_names = ExtractCustomerType(accent_raw_names)
    # accent_raw_names = accent_raw_names[accent_raw_names['customer_type'].isna() &
    #                                     (accent_raw_names['name'].str.split(' ').str.len() <= 5)][['name']]
    # accent_raw_names.columns = ['full_name']
    # accent_raw_names['source'] = 'raw'

    # # NAME NOT ACCENT
    # # Filter
    # old_names = dict_name_cdp['unidecode_full_name'].tolist()
    # process_names = not_accent_raw_names[~not_accent_raw_names['name'].isin(
    #     old_names)]

    # # Run: accent, split, gender
    # print(f'{len(process_names)} names')
    # process_names = process_enrich(
    #     process_names,
    #     name_col='name',
    #     n_cores=1
    # )
    # process_names = process_names[['name', 'predict']].drop_duplicates()
    # process_names['source'] = 'enrich accent'

    # # CONCAT: name accent + name not accent
    # pre_names = pd.concat([accent_raw_names, process_names], ignore_index=True)

    # # sparse name
    # with mp.Pool(8) as pool:
    #     pre_names[['last_name', 'middle_name', 'first_name']
    #               ] = pool.map(SplitName, pre_names['full_name'])
    # ? END CHANGE BLOCK

    # gender
    pre_names = process_predict_gender(
        pre_names,
        name_col='enrich_name',
        n_cores=1
    )
    # pre_names = Name2Gender(pre_names, name_col='full_name')
    # pre_names = pre_names.rename(columns={'predict_gender': 'gender'})

    # append
    pre_names['d'] = date_str
    dict_name_cdp = pd.concat([pre_names, dict_name_cdp], ignore_index=True)
    dict_name_cdp = dict_name_cdp.drop_duplicates(
        subset=['name'], keep='first')

    # # Post-process
    # # split data
    # dict_name_cdp['unidecode_full_name'] = dict_name_cdp['full_name'].apply(
    #     unidecode)
    # only_name = dict_name_cdp[dict_name_cdp['full_name'].str.split(
    #     ' ').str.len() == 1].copy()
    # full_name = dict_name_cdp[dict_name_cdp['full_name'].str.split(
    #     ' ').str.len() != 1].copy()

    # # post case only_name
    # only_name['unidecode_full_name'] = only_name['enrich_name']
    # only_name = only_name[only_name['unidecode_full_name'].str.lower().isin(
    #     word_name)]
    # only_name = only_name[only_name['source'] == 'enrich accent']

    # # concat + drop_dup
    # dict_name_cdp = pd.concat([only_name, full_name], ignore_index=True)
    # dict_name_cdp = dict_name_cdp.drop(columns=['unidecode_full_name'])

    # SAVE DEV
    dict_name_cdp.to_parquet(
        ROOT_PATH + '/utils/dict_name_latest.parquet', filesystem=hdfs, index=False)

    # SAVE PRODUCT
    pro_dict_name_cdp = dict_name_cdp.drop(columns=['source', 'd'])
    pro_dict_name_cdp = pro_dict_name_cdp.drop_duplicates().sample(frac=1)

    pro_dict_name_cdp.to_parquet(
        '/data/fpt/ftel/cads/dep_solution/sa/cdp/data/dict_name_latest.parquet', filesystem=hdfs, index=False)


# ENRICH NAME
def LoadNameGender(date_str, key='phone'):
    # create
    pre_path = ROOT_PATH + '/pre'
    cttv_s = ['fo', 'fplay', 'ftel', 'sendo', 'fshop', 'longchau']
    name_gender = pd.DataFrame()

    # loop
    for name_cttv in cttv_s:
        # read data
        data_cttv = pd.read_parquet(f'{pre_path}/{name_cttv}.parquet/d={date_str}', filesystem=hdfs,
                                    columns=[key, 'name', 'customer_type', 'gender'])

        # filters
        data_cttv = data_cttv[data_cttv[key].notna(
        ) & data_cttv['name'].notna()].copy()
        data_cttv['source_name'] = name_cttv.upper()

        # append
        name_gender = pd.concat([name_gender, data_cttv], ignore_index=True)

    # drop duplicate
    name_gender = name_gender.drop_duplicates(
        subset=[key, 'name', 'customer_type'], keep='first')
    name_gender = name_gender.rename(columns={'name': 'raw_name'})

    # load name (with accent)
    dict_name_cdp = pd.read_parquet(ROOT_PATH + '/utils/dict_name_latest.parquet',
                                    filesystem=hdfs, columns=['full_name', 'd', 'source'])
    dict_name_cdp['unidecode_full_name'] = dict_name_cdp['full_name'].apply(
        unidecode)
    dict_name_cdp = dict_name_cdp.sort_values(
        by=['unidecode_full_name', 'source'], ascending=False)
#     dict_name_cdp = dict_name_cdp.drop_duplicates(subset=['unidecode_full_name'], keep='last')
    dict_name_cdp = dict_name_cdp[dict_name_cdp['source'] == 'enrich accent'].drop_duplicates(
        subset=['unidecode_full_name'], keep='first')
    dict_name_cdp = dict_name_cdp[['unidecode_full_name', 'full_name']]
    dict_name_cdp.columns = ['raw_name', 'name_with_accent']

    # merge
    name_gender = name_gender.merge(dict_name_cdp, how='left', on=['raw_name'])
    name_gender.loc[name_gender['name_with_accent'].notna(
    ), 'name'] = name_gender['name_with_accent']
    name_gender.loc[name_gender['name'].isna(
    ), 'name'] = name_gender['raw_name']
    name_gender = name_gender.drop(columns=['name_with_accent'])

    # return
    return name_gender


def CoreBestName(raw_names_n, key='phone'):
    # Skip name (non personal)
    map_name_customer = raw_names_n[['raw_name']].copy().drop_duplicates()
    map_name_customer.columns = ['name']
    map_name_customer = ExtractCustomerType(map_name_customer)
    map_name_customer['num_word'] = map_name_customer['name'].str.split(
        ' ').str.len()
    skip_names = map_name_customer[map_name_customer['customer_type'].notna() |
                                   (map_name_customer['num_word'] > 4)]['name'].unique()

    skip_names_df = raw_names_n[raw_names_n['raw_name'].isin(
        skip_names)][[key, 'raw_name']].copy().drop_duplicates()
    names_df = raw_names_n[~raw_names_n['raw_name'].isin(
        skip_names)][[key, 'raw_name']].copy().drop_duplicates()

    print(">> Skip/Filter name")

    # Split name: last, middle, first
    map_split_name = names_df[['raw_name']].copy().drop_duplicates()
    with mp.Pool(8) as pool:
        map_split_name[['last_name', 'middle_name', 'first_name']
                       ] = pool.map(SplitName, map_split_name['raw_name'])
    names_df = names_df.merge(map_split_name, how='left', on=['raw_name'])

    # Create group_id
    names_df['unidecode_first_name'] = names_df['first_name'].apply(unidecode)
    names_df['group_id'] = names_df[key] + \
        '-' + names_df['unidecode_first_name']
    names_df = names_df.drop(columns=['unidecode_first_name'])

    # Split case process best_name
    names_df.loc[names_df['last_name'].notna(
    ), 'unidecode_last_name'] = names_df.loc[names_df['last_name'].notna(), 'last_name'].apply(unidecode)
    names_df['num_last_name'] = names_df.groupby(
        by=['group_id'])['unidecode_last_name'].transform('nunique')

    info_name_columns = ['group_id', 'raw_name',
                         'last_name', 'middle_name', 'first_name']
    names_n_df = names_df[names_df['num_last_name']
                          >= 2][info_name_columns].copy()
    names_1_df = names_df[names_df['num_last_name']
                          < 2][info_name_columns].copy()

    print(">> Create group_id")

    # Process case: 1 first_name - n last_name
    post_names_n_df = names_n_df[names_n_df['last_name'].isna()].copy()
    map_names_n_df = names_n_df[names_n_df['last_name'].notna() &
                                names_n_df['group_id'].isin(post_names_n_df['group_id'])].copy()

    map_names_n_df['num_char'] = map_names_n_df['raw_name'].str.len()
    map_names_n_df['num_word'] = map_names_n_df['raw_name'].str.split(
        ' ').str.len()
    map_names_n_df['accented'] = map_names_n_df['raw_name'] != map_names_n_df['raw_name'].apply(
        unidecode)
    map_names_n_df = map_names_n_df.sort_values(
        by=['group_id', 'num_word', 'num_char', 'accented'], ascending=False)
    map_names_n_df = map_names_n_df.groupby(by=['group_id']).head(1)
    map_names_n_df = map_names_n_df[['group_id', 'raw_name']].rename(
        columns={'raw_name': 'best_name'})

    post_names_n_df = post_names_n_df.merge(
        map_names_n_df, how='left', on=['group_id'])
    post_names_n_df = post_names_n_df[['group_id', 'raw_name', 'best_name']]

    names_n_df = names_n_df.merge(post_names_n_df, how='left', on=[
                                  'group_id', 'raw_name'])
    names_n_df.loc[names_n_df['best_name'].isna(
    ), 'best_name'] = names_n_df['raw_name']
    names_n_df = names_n_df[['group_id', 'raw_name', 'best_name']]

    print(">> 1 first_name - n last_name")

    # Process case: 1 first_name - 1 last_name
    map_names_1_df = names_1_df[['group_id']].drop_duplicates()
    for element_name in ['last_name', 'middle_name', 'first_name']:
        # filter data detail
        map_element_name = names_1_df[names_1_df[element_name].notna(
        )][['group_id', element_name]].copy().drop_duplicates()

        # create features
        map_element_name[f'unidecode_{element_name}'] = map_element_name[element_name].apply(
            unidecode)
        map_element_name['num_overall'] = map_element_name.groupby(
            by=['group_id', f'unidecode_{element_name}'])[element_name].transform('count')
        map_element_name = map_element_name.drop(
            columns=f'unidecode_{element_name}')

        map_element_name['num_char'] = map_element_name[element_name].str.len()
        map_element_name['num_word'] = map_element_name[element_name].str.split(
            ' ').str.len()
        map_element_name['accented'] = map_element_name[element_name] != map_element_name[element_name].apply(
            unidecode)

        # approach to choice best
        # map_element_name = map_element_name.sort_values(by=['group_id', 'num_overall', 'num_char', 'num_word', 'accented'], ascending=False)
        map_element_name = map_element_name.sort_values(
            by=['group_id', 'accented', 'num_overall', 'num_char', 'num_word'], ascending=False)
        map_element_name = map_element_name.groupby(by=['group_id']).head(1)
        map_element_name = map_element_name[['group_id', element_name]]
        map_element_name.columns = ['group_id', f'best_{element_name}']

        # merge
        map_names_1_df = map_names_1_df.merge(
            map_element_name, how='left', on=['group_id'])
        map_names_1_df.loc[map_names_1_df[f'best_{element_name}'].isna(
        ), f'best_{element_name}'] = None

    # combine element name
    dict_trash = {'': None, 'Nan': None, 'nan': None, 'None': None,
                  'none': None, 'Null': None, 'null': None, "''": None}
    columns = ['best_last_name', 'best_middle_name', 'best_first_name']
    map_names_1_df['best_name'] = map_names_1_df[columns].fillna('').agg(' '.join, axis=1).str.replace(
        '(?<![a-zA-Z0-9]),', '', regex=True).str.replace('-(?![a-zA-Z0-9])', '', regex=True)
    map_names_1_df['best_name'] = map_names_1_df['best_name'].str.strip().replace(
        dict_trash)
    map_names_1_df.loc[map_names_1_df['best_name'].isna(), 'best_name'] = None

    # merge
    names_1_df = names_1_df.merge(
        map_names_1_df[['group_id', 'best_name']], how='left', on=['group_id'])
    names_1_df = names_1_df[['group_id', 'raw_name', 'best_name']]

    print(">> 1 first_name - 1 last_name")

    # Concat
    names_df = pd.concat([names_1_df, names_n_df], ignore_index=True)

    # Calculate simility_score
    name_list_1 = list(names_df['raw_name'].unique())
    name_list_2 = list(names_df['best_name'].unique())
    map_element_name = pd.DataFrame()
    map_element_name['name'] = list(set(name_list_1) | set(name_list_2))

    with mp.Pool(8) as pool:
        map_element_name[['last_name', 'middle_name', 'first_name']] = pool.map(
            SplitName, map_element_name['name'])

    for flag in ['raw', 'best']:
        temp = map_element_name.copy()
        temp.columns = [f'{flag}_name', f'{flag}_last_name',
                        f'{flag}_middle_name', f'{flag}_first_name']
        names_df = names_df.merge(temp, how='left', on=[f'{flag}_name'])

    # similar score by element
    for element_name in ['last_name', 'middle_name', 'first_name']:
        # split data to compare
        condition_compare = names_df[f'raw_{element_name}'].notna(
        ) & names_df[f'best_{element_name}'].notna()
        compare_names_df = names_df[condition_compare].copy()
        not_compare_names_df = names_df[~condition_compare].copy()

        # compare raw with best
        compare_names_df[f'similar_{element_name}'] = compare_names_df[f'raw_{element_name}'].apply(
            unidecode) == compare_names_df[f'best_{element_name}'].apply(unidecode)
        compare_names_df[f'similar_{element_name}'] = compare_names_df[f'similar_{element_name}'].astype(
            int)

        not_compare_names_df[f'similar_{element_name}'] = 1

        # concat
        names_df = pd.concat(
            [compare_names_df, not_compare_names_df], ignore_index=True)

    weights = [0.25, 0.25, 0.5]
    names_df['simility_score'] = weights[0]*names_df['similar_last_name'] + weights[1] * \
        names_df['similar_middle_name'] + \
        weights[2]*names_df['similar_first_name']

    print(">> Simility_score")

    # Postprocess
    pre_names_df = names_df[['group_id', 'raw_name',
                             'best_name', 'simility_score']].copy()
    pre_names_df[key] = pre_names_df['group_id'].str.split('-').str[0]
    pre_names_df = pre_names_df.drop(columns=['group_id'])

    pre_names_df = pre_names_df.append(skip_names_df, ignore_index=True)
    pre_names_df.loc[pre_names_df['best_name'].isna(
    ), 'best_name'] = pre_names_df['raw_name']
    pre_names_df.loc[pre_names_df['simility_score'].isna(),
                     'simility_score'] = 1

    print(">> Postprocess")

    # Merge
    pre_names_n = raw_names_n.merge(
        pre_names_df, how='left', on=[key, 'raw_name'])
    pre_names_n.loc[pre_names_n['best_name'].isna(
    ), 'best_name'] = pre_names_n['raw_name']
    pre_names_n.loc[pre_names_n['simility_score'].isna(), 'simility_score'] = 1

    # Find source best_name
    pre_names_n['score_by_best'] = pre_names_n[['raw_name', 'best_name']].apply(lambda row: SequenceMatcher(None, row.raw_name, row.best_name).ratio(),
                                                                                axis=1)
    map_source_best_name = pre_names_n.sort_values(by=[key, 'best_name', 'score_by_best'],
                                                   ascending=False).groupby(by=[key, 'best_name']).head(1)[[key, 'best_name', 'source_name']].copy()
    map_source_best_name = map_source_best_name.rename(
        columns={'source_name': 'source_best_name'})
    pre_names_n = pre_names_n.merge(
        map_source_best_name, how='left', on=[key, 'best_name'])
    pre_names_n = pre_names_n.drop(columns=['score_by_best'])

    # Return
    return pre_names_n


def FindUniqueName(name_gender_by_key, date_str, key='phone'):
    print(">> Unique")
    # key have: duplicated vs only
    dup_key_s = name_gender_by_key[name_gender_by_key[key].duplicated(
    )][key].unique()
    dup_name_gender_by_key = name_gender_by_key[name_gender_by_key[key].isin(
        dup_key_s)].copy()
    only_name_gender_by_key = name_gender_by_key[~name_gender_by_key[key].isin(
        dup_key_s)].copy()

    # PROCESS DUPLICATED
    # params
    priority_names = {
        'ftel': ['name', 'active_date', 'last_active'],
        'sendo': ['name', 'active_date', 'last_active'],
        'fplay': ['name', 'active_date', 'last_active'],
        'fo': ['name', 'active_date', 'last_active'],
        'fshop': ['name', 'active_date', 'last_active'],
        'longchau': ['name', 'active_date', 'last_active']
    }
    raw_path = ROOT_PATH + '/raw'
    utils_path = ROOT_PATH + '/utils'

    raw_name = pd.DataFrame(
        columns=[key, 'raw_name', 'active_date', 'last_active', 'f_group', 'priority'])
    priority_index = 1
    for name_cttv, info_columns in priority_names.items():
        # print(name_cttv)

        # load data
        valid_key = pd.read_parquet(f'{utils_path}/valid_{key}_latest.parquet',
                                    filters=[
                                        (f'is_{key}_valid', '=', True), (key, 'in', dup_key_s)],
                                    columns=[f'{key}_raw', key], filesystem=hdfs)
        value_valid_key = valid_key[valid_key[f'{key}_raw'].notna(
        )][f'{key}_raw'].unique()
        data = pd.read_parquet(f'{raw_path}/{name_cttv}.parquet/d={date_str}',
                               filters=[(key, 'in', value_valid_key)],
                               columns=[key]+info_columns, filesystem=hdfs)

        data = data.rename(columns={key: f'{key}_raw', 'name': 'raw_name'})
        data = data.merge(valid_key, how='inner', on=[f'{key}_raw'])
        data = data.drop(columns=[f'{key}_raw'])

        # add key
        data = data[data[key].notna()]
        data['f_group'] = name_cttv.upper()
        data['priority'] = priority_index

        # append
        raw_name = pd.concat([raw_name, data], ignore_index=True)
        priority_index += 1

    # process name
    map_name = raw_name[['raw_name']].copy().drop_duplicates()

    # clean name
    map_name['clean_name'] = map_name.apply(
        lambda row: CleanName(row['raw_name'], ''), axis=1)
    map_name = map_name[map_name['clean_name'].notna()]

    # format name
    dict_trash = {'': None, 'Nan': None, 'nan': None, 'None': None,
                  'none': None, 'Null': None, 'null': None, "''": None}
    with mp.Pool(8) as pool:
        map_name[['last_name', 'middle_name', 'first_name']
                 ] = pool.map(SplitName, map_name['clean_name'])
    element_name_columns = ['last_name', 'middle_name', 'first_name']
    map_name['clean_name'] = map_name[element_name_columns].fillna('').agg(' '.join, axis=1).str.replace(
        '(?<![a-zA-Z0-9]),', '', regex=True).str.replace('-(?![a-zA-Z0-9])', '', regex=True)
    map_name['clean_name'] = map_name['clean_name'].str.strip().replace(dict_trash)
    map_name = map_name.drop(columns=element_name_columns)

    map_name_active = raw_name.merge(map_name, how='left', on=['raw_name'])
    map_name_active['raw_name'] = map_name_active['clean_name']
    map_name_active = map_name_active.drop(
        columns=['clean_name', 'f_group']).drop_duplicates()

    # merge key
    dup_name_gender_by_key = dup_name_gender_by_key.merge(
        map_name_active, how='left', on=[key, 'raw_name'])
    stats_best_name = dup_name_gender_by_key.groupby(by=[key, 'best_name'])[
        'best_name'].agg(num_overall='count').reset_index()
    dup_name_gender_by_key = dup_name_gender_by_key.merge(
        stats_best_name, how='left', on=[key, 'best_name'])
#     dup_name_gender_by_key = dup_name_gender_by_key.sort_values(by=[key, 'num_overall', 'priority', 'last_active', 'active_date'],
#                                                                   ascending=[True, False, True, False, False])
    dup_name_gender_by_key = dup_name_gender_by_key.sort_values(by=[key, 'last_active', 'active_date', 'num_overall', 'priority'],
                                                                ascending=[True, False, False, False, True])

    # split: company vs personal
    condition_split = None
    if key == 'phone':
        condition_split = (dup_name_gender_by_key[key].str.len() == 11) | (
            dup_name_gender_by_key['customer_type'] != 'Ca nhan')

    elif key == 'email':
        condition_split = (
            dup_name_gender_by_key['customer_type'] != 'Ca nhan')

    com_dup_name_gender_by_key = dup_name_gender_by_key[condition_split].copy()
    per_dup_name_gender_by_key = dup_name_gender_by_key[~dup_name_gender_by_key[key].isin(
        com_dup_name_gender_by_key[key].unique())].copy()

    # unique company
    com_dup_name_gender_by_key = com_dup_name_gender_by_key.drop(
        columns=['active_date', 'last_active', 'priority', 'num_overall'])
    com_dup_name_gender_by_key['num_word'] = com_dup_name_gender_by_key['raw_name'].str.split(
        ' ').str.len()
    com_dup_name_gender_by_key['num_char'] = com_dup_name_gender_by_key['raw_name'].str.len(
    )
    com_dup_name_gender_by_key = com_dup_name_gender_by_key.sort_values(
        by=[key, 'num_word', 'num_char'], ascending=False)

    com_unique_name_gender_by_key = com_dup_name_gender_by_key.groupby(
        by=[key]).head(1)[[key, 'best_name', 'source_best_name']].drop_duplicates()
    com_unique_name_gender_by_key.columns = [
        key, 'unique_name', 'source_unique_name']

    # unique personal
    per_dup_name_gender_by_key['num_word'] = per_dup_name_gender_by_key['raw_name'].str.split(
        ' ').str.len()
    per_dup_name_gender_by_key['num_char'] = per_dup_name_gender_by_key['raw_name'].str.len(
    )
    per_dup_name_gender_by_key = per_dup_name_gender_by_key.sort_values(by=[key, 'last_active', 'active_date', 'num_overall', 'priority', 'num_word', 'num_char'],
                                                                        ascending=[True, False, False, False, True, False, False])
    per_dup_name_gender_by_key = per_dup_name_gender_by_key.groupby(
        by=[key]).head(1)[[key, 'best_name', 'source_best_name']].drop_duplicates()
    per_dup_name_gender_by_key.columns = [
        key, 'unique_name', 'source_unique_name']

    # concat unique key
    unique_name_gender_by_key = pd.concat(
        [com_unique_name_gender_by_key, per_dup_name_gender_by_key], ignore_index=True)

    # merge key
    dup_name_gender_by_key = dup_name_gender_by_key.drop(
        columns=['active_date', 'last_active', 'priority', 'num_overall']).drop_duplicates()
    dup_name_gender_by_key = dup_name_gender_by_key.merge(
        unique_name_gender_by_key, on=[key], how='left')

    # PROCESS ONLY
    only_name_gender_by_key['unique_name'] = only_name_gender_by_key['best_name']
    only_name_gender_by_key['source_unique_name'] = only_name_gender_by_key['source_name']

    # OUTPUT
    name_gender_by_key = pd.concat(
        [dup_name_gender_by_key, only_name_gender_by_key], ignore_index=True)
#     name_gender_by_key.columns = [key, 'raw_name', 'customer_type', 'gender', 'source_name', 'enrich_name',
#                                    'simility_score_enrich_name', 'source_enrich_name', 'enrich_gender',
#                                    'best_name', 'source_best_name']

    # rename
    name_gender_by_key = name_gender_by_key.drop(
        columns=['name']).rename(columns={'raw_name': 'name'})

    # return
    return name_gender_by_key


def PipelineBestName(date_str, key='phone'):
    # params
    utils_path = ROOT_PATH + '/utils'

    # load data
    raw_names = LoadNameGender(date_str, key)

    # rename
    raw_names = raw_names.rename(columns={'raw_name': 'original_name'}).rename(
        columns={'name': 'raw_name'})

    # split data to choise best name
    dup_ids = raw_names[raw_names[key].duplicated()][key].unique()
    raw_names_n = raw_names[raw_names[key].isin(dup_ids)].copy()
    raw_names_1 = raw_names[~raw_names[key].isin(dup_ids)].copy()

    # run pipeline best_name
    pre_names_n = CoreBestName(raw_names_n, key)

    # fake best_name
    pre_names_1 = raw_names_1.copy()
    pre_names_1['best_name'] = pre_names_1['raw_name']
    pre_names_1['simility_score'] = 1
    pre_names_1['source_best_name'] = pre_names_1['source_name']

    # concat
    pre_names = pd.concat([pre_names_1, pre_names_n], ignore_index=True)

    # best gender
    map_name_gender = pre_names[['best_name']].copy().drop_duplicates()
    map_name_gender = Name2Gender(map_name_gender, name_col='best_name')
    pre_names = pre_names.merge(map_name_gender, how='left', on=['best_name'])
    pre_names.loc[pre_names['gender'] == pre_names['predict_gender'],
                  'best_gender'] = pre_names['gender']
    pre_names.loc[pre_names['best_gender'].isna(), 'best_gender'] = None
    pre_names = pre_names.drop(columns=['predict_gender'])

    # rename
    pre_names = pre_names.rename(columns={'raw_name': 'name'}).rename(
        columns={'original_name': 'raw_name'})

    # unique name
    name_gender_by_key = FindUniqueName(pre_names, date_str, key)

    # return
    return name_gender_by_key


def DictNameGender():
    utils_path = ROOT_PATH + '/utils'

    # Load data CDP Profle
    data_name_cdp = pd.DataFrame()
    for key in ['phone', 'email']:
        print(key)
        data = pd.read_parquet(f'{utils_path}/name_by_{key}_latest.parquet', filesystem=hdfs,
                               columns=['name', 'gender', 'source_name', 'best_name', 'best_gender', 'source_best_name'])
        data_name_cdp = pd.concat([data_name_cdp, data], ignore_index=True)

    # rename columns
    data_name_cdp.columns = ['name', 'gender', 'source_name',
                             'enrich_name', 'enrich_gender', 'source_enrich_name']

    # process: concate (vertical)
    data_name_cdp_pre = data_name_cdp[['name', 'gender', 'source_name']].copy()
    data_name_cdp_pre.columns = ['name', 'gender', 'source_name']

    data_name_cdp_enrich = data_name_cdp[[
        'enrich_name', 'enrich_gender', 'source_enrich_name']].copy()
    data_name_cdp_enrich.columns = ['name', 'gender', 'source_name']

    data_name_cdp = pd.concat(
        [data_name_cdp_pre, data_name_cdp_enrich], ignore_index=True)
    data_name_cdp.loc[data_name_cdp['gender'].notna(
    ), 'source_gender'] = data_name_cdp['source_name']
    data_name_cdp = data_name_cdp.drop_duplicates()

    # process: drop dup
    data_name_cdp_gender = data_name_cdp[data_name_cdp['gender'].notna()].copy(
    )
    data_name_cdp_non_gender = data_name_cdp[data_name_cdp['gender'].isna() &
                                             ~data_name_cdp['name'].isin(data_name_cdp_gender['name'].unique())].copy()
    data_name_cdp = pd.concat([data_name_cdp_gender, data_name_cdp_non_gender],
                              ignore_index=True).drop_duplicates(subset=['name'], keep='first')

    # Load data Email
    name_email = pd.read_parquet('/data/fpt/ftel/cads/dep_solution/sa/cdp/data/email_dict_latest.parquet',
                                 filesystem=hdfs, filters=[('username_iscertain', '=', True)])[['username']]
    name_email = name_email[name_email['username'].notna()].drop_duplicates()
    name_email.columns = ['name']
    name_email = name_email[~name_email['name'].isin(data_name_cdp['name'])]
    name_email['source_name'] = 'Email'

    # Concat
    data_name_cdp = pd.concat([data_name_cdp, name_email], ignore_index=True)

    # Model name2gender
    data_name_cdp = Name2Gender(data_name_cdp, name_col='name')
    data_name_cdp.loc[data_name_cdp['gender'].isna(
    ), 'gender'] = data_name_cdp['predict_gender']

    # PostProcess
    customer_name = ExtractCustomerType(data_name_cdp)
    com_name = customer_name[customer_name['customer_type'].notna() |
                             (customer_name['name'].str.split(' ').str.len() > 5)]['name'].unique()
    data_name_cdp.loc[data_name_cdp['name'].isin(com_name), 'gender'] = None
    data_name_cdp.loc[data_name_cdp['gender'].notna(
    ) & data_name_cdp['source_gender'].isna(), 'source_gender'] = 'Name2Gender'
    data_name_cdp = data_name_cdp.drop(
        columns=['name_en', 'name_len', 'predict_gender'])

    # Save
    data_name_cdp.to_parquet(
        f'{utils_path}/track_name_gender_latest.parquet', filesystem=hdfs, index=False)
