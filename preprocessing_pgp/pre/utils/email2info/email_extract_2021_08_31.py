import numpy as np
from math import log
import os
import os.path
from os import path
import sys
import glob
import re
from tqdm import tqdm
import traceback
import unidecode
import itertools
import json
import pickle


import datetime
import time
from datetime import date
from dateutil.relativedelta import relativedelta

import multiprocessing
from multiprocessing import Pool
from functools import partial

import pandas as pd

import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

# ADD SPECIAL CASE OR REMOVE OUTLIER CASE IN AVAILABLE SOURCE.
SPEC_LAST_NAME = ['Nguyễn Văn', 'Nguyễn Thị', 'Ngô']
SKIP_LAST_NAME = ['Ánh', 'Trang', 'Nhữ', 'Bao', 'Cổ', 'Y', 'Thi', 'Giang', 'Vi', 'Viên', 'La', 'Điểu', 'GiĂĄp', 'Chaâu', 'Laâm', 'Löu', 'Höùa', 'Vuõ',
                'Ất', 'Lư', 'Lộ', 'Kỷ', 'Sử', 'Xa', 'Vy', 'Kỳ', 'Bá', 'Sơ','Lò', 'Y-', 'Ma', 'Cà', 'Lỡ', 'Jơ', 'Student']
SKIP_FIRST_NAME = ['Nhieu']

# 3rd-Party Data Source
LAST_NAME_JSON = '/home/khanhhb3/CDP/ADHOC/2021-08/3rdParty/vietnamese-namedb/vn_name_db.json'
BOY_NAME_DB = '/home/khanhhb3/CDP/ADHOC/2021-08/3rdParty/vietnamese-namedb/bnames_db.txt'
GIRL_NAME_DB = '/home/khanhhb3/CDP/ADHOC/2021-08/3rdParty/vietnamese-namedb/gnames_db.txt'
LOCDICT_VN_PATH = '/home/khanhhb3/CDP/ADHOC/2021-08/Data/AddressAdhoc/dvhc_dict.parquet' 
WORDNAME_DF = '/bigdata/fdp/bk/fo/ad-hoc/gender_dectection/dataset/staging/word_name.parquet.gzip'
GENDER_PRED_MODEL = '/bigdata/fdp/user/namdp11/name2gender/not_accented/logistic_pipeline.pkl'
EMAIL_GROUP_DICT = '/bigdata/fdp/user/khanhhb3/adhoc/2021/email_adhoc_2021-09/email_group_dict_2021-11-30.parquet'

class EmailDataPath:
    def __init__(self, data_source):
        # data_source: dir for path config file instead of default dir.
        if data_source == 'default':
            self.lnme_json = LAST_NAME_JSON
            self.bnme_db = BOY_NAME_DB
            self.gnme_db = GIRL_NAME_DB
            self.loc_db = LOCDICT_VN_PATH
            self.wordname_df = WORDNAME_DF
            self.gpred_model = GENDER_PRED_MODEL
            self.group_dict = EMAIL_GROUP_DICT
        else:
            if path.exists(data_source):
                config_path = json.load(open(data_source, 'r'))
                for item in config_path['ref_dir']:
                    if item['name'] == 'lastname_json':
                        self.lnme_json = item['dir']
                    elif item['name'] == 'boy_name_db':
                        self.bnme_db = item['dir']
                    elif item['name'] == 'girl_name_db':
                        self.gnme_db = item['dir']
                    elif item['name'] == 'loc_vn_dict':
                        self.loc_db = item['dir']
                    elif item['name'] == 'wordname_df':
                        self.wordname_df = item['dir']
                    elif item['name'] == 'gpred_model':
                        self.gpred_model = item['dir']
                    elif item['name'] == 'group_dict':
                        self.group_dict = item['dir']
            else:
                raise ValueError("Path config file not exist !")
                

class Email2Username:
    def __init__(self, path_obj):
        self.fullname_dict = json.load(open(path_obj.lnme_json, 'r'))
        self.bnames_db = pd.read_csv(path_obj.bnme_db, header = None)
        self.gnames_db = pd.read_csv(path_obj.gnme_db, header = None)
        
        self.name_list = None
        self.name_w_acc = None
        self.lname_list = None
        self.fname_list = None
        self.wordcost = None
        self.maxword = None
        
    def collect_last_name(self, last_name_list):
        spec_last_name = SPEC_LAST_NAME
        lname_collect = last_name_list + spec_last_name
        df_lname = pd.DataFrame({'last_name':lname_collect})
        lname_list_fin = sorted(df_lname['last_name'].unique().tolist(), reverse=True)
        
        return lname_list_fin
    
    def collect_first_name(self, df_fname):
        df_fn_c = df_fname[0].astype(str).str.split(' ', expand=True)
        df_fnames = pd.concat([df_fname[0], df_fn_c[0], df_fn_c[1]]).drop_duplicates()
        fname_list = df_fnames.unique().tolist()
        
        return fname_list
    
    def collect_name_dict(self):
        # Collect Vietnamese first names and last names from different source.
        fullname_dict = self.fullname_dict        
        skip_last_name = SKIP_LAST_NAME
        last_name_list = []
        for name in fullname_dict:
            if name['last_name_group'] in skip_last_name:
                continue
            last_name_list.append(name['last_name_group'])
        lname_list_o = self.collect_last_name(last_name_list)
        
        df_boy = self.bnames_db
        df_girl = self.gnames_db
        df_fname = pd.concat([df_boy, df_girl]).reset_index(drop=True)
        df_fname = df_fname[~df_fname[0].isin(SKIP_FIRST_NAME)]
        fname_list_o = self.collect_first_name(df_fname.copy())
        
        df_names = pd.DataFrame({'name': lname_list_o+fname_list_o})
        df_names['length'] = df_names['name'].str.len()
        df_names = df_names.sort_values(by='length', ascending=False)
        df_names['name'] = df_names['name'].apply(unidecode.unidecode)
        df_names['name_clean'] = df_names['name'].str.lower().str.replace(' ','')
        
        self.fname_list = pd.Series(fname_list_o).apply(unidecode.unidecode).str.lower().str.replace(' ','')
        self.lname_list = pd.Series(lname_list_o).apply(unidecode.unidecode).str.lower().str.replace(' ','')
        self.name_list = df_names['name_clean'].unique().tolist()
        self.name_w_acc = pd.Series(df_names['name'].values, index = df_names['name_clean']).to_dict()

        return 1
    
    def build_ncost_dict(self):
        # Build a cost dictionary, assuming Zipf's law and cost = -math.log(probability).

        if self.name_list is None:
            return -1
        self.wordcost = dict((k, log((i+1)*log(len(self.name_list)))) for i,k in enumerate(self.name_list))
        self.maxword = max(len(x) for x in self.name_list)
        
        return 1
    
    def create_names_dict(self):
        # Build all possible format of date of birth (only containing year of birth case).
        if self.collect_name_dict() and  self.build_ncost_dict():
            return 1
        else:
            raise NotImplementedError('Cannot build Names dictionary!')
        
        
    def extract_name_component(self, s):
        """Uses dynamic programming to infer the location of spaces in a string
        without spaces."""
        # Find the best match for the i first characters, assuming cost has
        # been built for the i-1 first characters.
        # Returns a pair (match_cost, match_length).
        if self.wordcost is None or self.maxword is None:
            raise ValueError("Cost Dict of names NA !")
        wordcost = self.wordcost
        maxword = self.maxword
        
        def best_match(i):
            candidates = enumerate(reversed(cost[max(0, i-maxword):i]))
            return min((c + wordcost.get(s[i-k-1:i], 9e999), k+1) for k,c in candidates)

        # Build the cost array.
        cost = [0]
        for i in range(1,len(s)+1):
            c,k = best_match(i)
            cost.append(c)

        # Backtrack to recover the minimal-cost string.
        out = []
        i = len(s)
        while i>0:
            c,k = best_match(i)
            assert c == cost[i]
            out.append(s[i-k:i])
            i -= k

        return " ".join(reversed(out))
    

    def get_username(self, df, key_col, input_col, merge_col = 'key'):
        """Use built dictionary to extract name component from email name and then reorder based on rule-based."""
        # Find the best match for the names component in each dataframe row,
        # assuming dict has been built for all names substring.
        # Returns a DataFrame.
        norm_name_dict = self.name_w_acc
        lastname_list = self.lname_list
        firstname_list = self.fname_list
        
        if norm_name_dict is None or lastname_list is None or firstname_list is None:
            return -1
        
        df[input_col] = df[input_col].str.replace('.','')
        df['username_candidate'] = df.apply(lambda x: self.extract_name_component(x[input_col]), axis=1)
        df['username'] = ''
        df['last_name_fill'] = False
        df['first_name_found'] = np.nan
        test = df['username_candidate'].astype(str).str.split(' ', expand=True)
        for col in test.columns:
            username_valid = test[col].str.len()>1
            lastname_condition = ((test[col].isin(lastname_list)))
            #Find lastname for the first time
            lastname_case_1 = username_valid & lastname_condition & ((df['username'].str.len() == 0) | (df['last_name_fill'] == False))
            #Find another last name
            lastname_case_2 = username_valid & lastname_condition & (df['username'].str.len() != 0) & (df['last_name_fill'] == True)

            firstname_condition = (test[col].isin(firstname_list)) & (~lastname_condition)
            #Find firstname before lastname found
            firstname_case_1 = username_valid & firstname_condition & (df['last_name_fill']== False)
            #Find firstname after lastname
            firstname_case_2 = username_valid & firstname_condition & (df['last_name_fill']== True)
            test.loc[username_valid, col] = test.loc[username_valid, col].map(norm_name_dict)
            df.loc[lastname_case_1, 'username'] = test.loc[lastname_case_1, col] + df.loc[lastname_case_1, 'username']
            df.loc[lastname_case_2, 'username'] +=  ' ' + test.loc[lastname_case_2, col]
            df.loc[lastname_case_1, 'last_name_fill'] = True
            df.loc[firstname_case_1 & (df['first_name_found'].notna()), 'first_name_found'] += ' ' + test.loc[firstname_case_1 & (df['first_name_found'].notna()),col]
            
            df.loc[firstname_case_1 & (df['first_name_found'].isna()), 'first_name_found'] = test.loc[firstname_case_1 & (df['first_name_found'].isna()), col]
            df.loc[firstname_case_2, 'username'] = df.loc[firstname_case_2, 'username'] + ' ' + test.loc[firstname_case_2, col]
        df.loc[df['first_name_found'].notna(), 'username'] += ' ' + df.loc[df['first_name_found'].notna(), 'first_name_found']
        df['username'] = df['username'].astype(str).str.strip()
        df.loc[df['username'].astype(str).str.len() <= 1, 'username'] = np.nan
        return df[[merge_col, key_col, input_col, 'username']]
    
class Email2YearOfBirth:
    def __init__(self):
        self.dnum_list = None
        self.dpat_list = None
        self.dform_list = None
    
    def create_yob_dict(self):
        # Build all possible format of date of birth (only containing year of birth case).
        # Update year of birth from > 1970 to > 1950 (29/08/2022)
        fullyear_pat = '(19[5-9][0-9]|200[0-9])'
        halfyear_pat = '([7-9][0-9])'
        fullmonth_pat = '(0[1-9]|1[012])'
        halfmonth_pat = '([1-9])'
        nonemonth_pat = ''
        fullday_pat = '(0[1-9]|[12][0-9]|3[01])'
        halfday_pat = '([1-9])'
        noneday_pat = ''

        number_digits_dict = {'fullday':2, 'halfday':1, 'noneday':0, 'fullmonth':2, 'halfmonth':1, 'nonemonth':0, 'fullyear':4, 'halfyear':2}
        pat_dict = {'fullday':fullday_pat, 'halfday':halfday_pat, 'noneday':noneday_pat, 
                    'fullmonth':fullmonth_pat, 'halfmonth':halfmonth_pat, 'nonemonth':nonemonth_pat, 
                    'fullyear':fullyear_pat, 'halfyear':halfyear_pat}
        format_dict = {'fullday':'dd', 'halfday':'d', 'noneday':'', 'fullmonth':'mm', 'halfmonth':'m', 'nonemonth':'', 'fullyear':'yyyy', 'halfyear':'yy'}

        day_year_list = [['fullyear','halfyear'], ['fullday','halfday'],['fullmonth','halfmonth']]
        combination_days_year_list = list(itertools.product(*day_year_list))
        combination_days_year_list.append(('fullyear', 'noneday', 'nonemonth'))
        combination_days_year_list.append(('halfyear', 'noneday', 'nonemonth'))
        combination_days_year_list.append(('halfyear', 'noneday', 'fullmonth'))
        combination_days_year_list.append(('halfyear', 'noneday', 'halfmonth'))
        combination_days_year_list.append(('fullyear', 'noneday', 'fullmonth'))
        combination_days_year_list.append(('fullyear', 'noneday', 'halfmonth'))
        # remove case abc2580@gmail.com -> Not yob:1980
        combination_days_year_list.remove(('halfyear', 'halfday', 'halfmonth'))
        
        num_digits_list = []
        pat_list = []
        format_list = []

        day = None
        month = None
        year = None
        for item in combination_days_year_list:
            year = item[0]
            day = item[1]
            month = item[2]
            num_digits_list.append(number_digits_dict[day] + number_digits_dict[month] + number_digits_dict[year])
            pat_list.append(pat_dict[day] + pat_dict[month] + pat_dict[year])  
            format_list.append(format_dict[day] + format_dict[month] + format_dict[year])        
        
        self.dnum_list = num_digits_list
        self.dpat_list = pat_list
        self.dform_list = format_list
        
        return 1
    
    def yearofbirth_match(self, df, input_col, pattern, format_datetime):
        """Uses built pattern substring for yob extraction."""
        # Find match for built pattern.
        # Returns a df with filled columns (year_of_birth, format_yob).
        extraction = df[input_col].astype(str).str.extract(pattern)
        for col in extraction.columns[::-1]:
            condition = df['year_of_birth'].isna()
            df.loc[condition, 'year_of_birth'] = extraction[col]
            df.loc[condition, 'format_yob'] = format_datetime
        return df.copy()
    
    def get_yearofbirth(self, df, key_col, input_col, merge_col = 'key'):
        """Uses built method to implement a df of emails."""
        # Find YOB in each row.
        # Returns a df with columns (year_of_birth, format_yob).
        num_digits_list = self.dnum_list
        pat_list = self.dpat_list
        format_list = self.dform_list
        if num_digits_list is None or pat_list is None or format_list is None:
            raise ValueError("Cost Dict of names NA !")
            
        df_yob_data = df[[merge_col, key_col, input_col]]
        df_yob_data['year_of_birth'] = np.nan
        df_yob_data['format_yob'] = np.nan

        df_yob_data['digits'] = df_yob_data['email_name'].apply(lambda x: sum(character.isdigit() for character in x))
        
        for num, pat, format_date in zip(num_digits_list, pat_list, format_list):
            map_dict = None
            df_map = None
            condition_yob = (df_yob_data['digits'] == num) & (df_yob_data['year_of_birth'].isna())
            df_map = self.yearofbirth_match(df_yob_data.loc[condition_yob].copy(), input_col, pat, format_date)
            df_map = df_map[df_map['year_of_birth'].notna()]
#             df_yob_data = pd.concat([df_yob_data, pd.DataFrame({'year_of_birth':df_map['year_of_birth'].values, 'format_yob':df_map['format_yob'].values},
#                                                                index = df_map[input_col])])
            yob_dict = dict(zip(df_map[key_col], df_map['year_of_birth']))
    
            df_yob_data.loc[df_yob_data['year_of_birth'].isna(),
                                        'year_of_birth']= df_yob_data.loc[df_yob_data['year_of_birth'].isna(),
                                                                                      key_col].map(yob_dict)
            
            df_yob_data['year_of_birth'] = df_yob_data['year_of_birth'].astype(float)
            df_yob_data.loc[df_yob_data['year_of_birth'] < 100, 'year_of_birth'] += 1900
        return df_yob_data[[merge_col, key_col, input_col, 'year_of_birth']]
    
class Email2Phone:
    def get_phone_number(self, df, key_col, input_col, merge_col = 'key'):
        """Uses regex to extract phone string."""
        # Find substring matching regex of phone string.
        # Returns a df with filled column (Phone).
        df['phone'] = df[input_col].str.extract('((0|84)[3|5|7|8|9|16|12][0-9]{8})')[0]
        return df[[merge_col, key_col, input_col, 'phone']].copy()

class Email2Address:
    def __init__(self, path_obj):
        self.path_obj = path_obj
        self.df_dvhc = None
    
    def get_norm_location_name(self, df_dvhc_dict):
        # Collect non accent VN location name (Ex. 'Tinh Nghe An',...).
        # Level 1
        df_dvhc_dict['DVHC_LV1_NORM_NAME'] = 'Unknown'
        condition1 = df_dvhc_dict['DVHC_LV1_NORM'].str.contains(r'\bTinh')
        df_dvhc_dict.loc[condition1,'DVHC_LV1_NORM_NAME'] = df_dvhc_dict.loc[condition1,'DVHC_LV1_NORM'].str.extract(r'\s(.*)')[0]
        df_dvhc_dict.loc[~condition1,'DVHC_LV1_NORM_NAME'] = df_dvhc_dict.loc[~condition1]['DVHC_LV1_NORM'].str.extract(r'\s\S*\s(.*)')[0]

        # Level 2
        df_dvhc_dict['DVHC_LV2_NORM_NAME'] = 'Unknown'
        condition1 = df_dvhc_dict['DVHC_LV2_NORM'].str.contains(r'^Quan|^Huyen') 
        condition2 = condition1 & df_dvhc_dict['DVHC_LV2_NORM'].str.contains(r'\b [0-1]|[0-9]\b')

        df_dvhc_dict.loc[condition1,'DVHC_LV2_NORM_NAME'] = df_dvhc_dict.loc[condition1,'DVHC_LV2_NORM'].str.extract(r'\s(.*)')[0]
        df_dvhc_dict.loc[condition2,'DVHC_LV2_NORM_NAME'] = df_dvhc_dict.loc[condition1,'DVHC_LV2_NORM']

        df_dvhc_dict.loc[~condition1,'DVHC_LV2_NORM_NAME'] = df_dvhc_dict.loc[~condition1]['DVHC_LV2_NORM'].str.extract(r'\s\S*\s(.*)')[0]
        return df_dvhc_dict.copy()
    
    def get_acron_location_name(self, df_dvhc_dict):
        # Collect acronym VN location name (Ex. 'na'-(Tinh Nghe An),...).
        s = df_dvhc_dict['DVHC_LV1_NORM_NAME'].str.lower()
        df_dvhc_dict['DVHC_LV1_ACRON_NAME'] = pd.Series(["".join([(y[0] if y[0] != '-' else '') for y in x.split()]) for x in s.tolist()])

        s = df_dvhc_dict['DVHC_LV2_NORM_NAME'].str.lower()
        ar_list = []
        for x in s.tolist():
            item_list = []
            for y in x.split():
                if y.isdigit():
                    item_list.append(y)
                else:
                    if y[0] != '-':
                        item_list.append(y[0])
            ar_list.append(''.join(item_list))
        df_dvhc_dict['DVHC_LV2_ACRON_NAME'] = pd.Series(ar_list)

        return df_dvhc_dict.copy()
    
    def create_address_dict(self):
        # Build VN Location format for Level 1 and Level 2.
        # Column DVHC_LV1_NORM_NAME (Ex. 'Tinh Nghe An',...).
        # Column DVHC_LV1_ACRON_NAME (Ex. 'na',...).
        df_dvhc_dict = pd.read_parquet(self.path_obj.loc_db)
        self.df_dvhc = df_dvhc_dict
        df_dvhc_dict_norm = self.get_norm_location_name(df_dvhc_dict.copy())
        df_dvhc_dict_full = self.get_acron_location_name(df_dvhc_dict_norm.copy())
        self.df_dvhc = df_dvhc_dict_full.copy()
        return 1
    def get_province_ref_dict(self, df_dvhc_ref):
        #LV1 Dictionary
        lv1_list = []
        #Explicit case
        col = 'DVHC_LV1_NORM_NAME'
        for count, item in df_dvhc_ref.iterrows():
                itemLower = item[col].lower().replace(' ','')
                if itemLower not in lv1_list:
                    lv1_list.append(itemLower)

        col = 'DVHC_LV1_ACRON_NAME'
        dup_acron_lv1 = df_dvhc_ref['DVHC_LV2_ACRON_NAME'].to_list()
        for count, item in df_dvhc_ref[['DVHC_LV1_NORM_NAME', 'DVHC_LV1_ACRON_NAME']].drop_duplicates().iterrows():
                itemLower = item[col].lower().replace(' ','')
                if itemLower in lv1_list:
                    if itemLower not in dup_acron_lv1:
                        dup_acron_lv1.append(itemLower)
                    lv1_list.remove(itemLower)
                elif itemLower not in lv1_list:
                    if itemLower not in dup_acron_lv1:
                        lv1_list.append(itemLower)
                        dup_acron_lv1.append(itemLower)

        return lv1_list.copy()

    def get_address(self, df, key_col, input_col, merge_col = 'key'):
        """Uses built VN location dict to extract location in email."""
        # Returns a df with filled column (Address).
        if self.df_dvhc is None:
            self.create_address_dict()
        df_dvhc = self.df_dvhc
        province_list = self.get_province_ref_dict(df_dvhc.copy())
        
        df_dvhc['LV1_MAP'] = df_dvhc['DVHC_LV1_NORM_NAME'].str.lower().str.replace(' ','')
        dvhc_lv1_add = df_dvhc['LV1_MAP'].to_list() + df_dvhc['DVHC_LV1_ACRON_NAME'].to_list()
        dvhc_lv1_norm_add = df_dvhc['DVHC_LV1_NORM'].to_list() + df_dvhc['DVHC_LV1_NORM'].to_list()
        df_norm_address = pd.DataFrame({'LV1_ADDRESS':dvhc_lv1_add, 'LV1_NORM': dvhc_lv1_norm_add})
        df_norm_address = df_norm_address.drop_duplicates().reset_index(drop=True)
        norm_address = pd.Series(df_norm_address['LV1_NORM'].values, index = df_norm_address.LV1_ADDRESS).to_dict()
        
        pat_pronvince_add = '\\b|'.join(r"{}".format(x) for x in province_list) + '\\b'
        address_province_extract = df[input_col].str.extractall('(?=('+ pat_pronvince_add + '))').unstack(-1).droplevel(0, axis=1)
        df['address'] = np.nan
        if address_province_extract.shape[0] != 0:
            address_province_extract['address'] = address_province_extract[0].map(norm_address)
            df['address'] = address_province_extract['address']

        return df[[merge_col, key_col, input_col, 'address']].copy()

class Email2Gender:
    def __init__(self, path_obj):
        self.path_obj = path_obj
        self.df_dvhc = None
    
    def load_model(self, file_path, module='pickle'):
        if module == 'pickle':
            model = pickle.load(open(file_path, 'rb'))
        return model

    def preprocess_fromName(self, df, name_col='username', max_length=None):
        df = df.copy()
        df['CleanName'] = df[name_col].str.lower().str.replace(
        r'\d', ' ').str.replace(
        rf'[{string.punctuation}]', ' ').str.replace(
        r'\s+', ' ').str.strip()

        return df

    def get_gender(self, df, key_col, input_col, merge_col = 'key', name_col='username'):
        prep_df = self.preprocess_fromName(df.copy(), name_col)
        pipeline = self.load_model(self.path_obj.gpred_model)
        
        predictions = pipeline.predict(prep_df['CleanName'].values)
        predictions = list(map(lambda x: 'M' if x == 1 else 'F', predictions))

        df['gender'] = pd.Series(predictions)
        return df[[merge_col, key_col, input_col, name_col, 'gender']]

class EmailCDP:
    def __init__(self, data_source='default'):
        self.dict_built = False
        self.path_list = EmailDataPath(data_source)
        self.email_uname = Email2Username(self.path_list)
        self.email_yob = Email2YearOfBirth()
        self.email_phone = Email2Phone()
        self.email_address = Email2Address(self.path_list)
        self.email_gender = Email2Gender(self.path_list)
        self.info_list={'username':self.extract_username, 'yob':self.extract_yob, 'phone':self.extract_phone,
                        'address':self.extract_address, 'gender':self.extract_gender, 'group':self.extract_group, 'automail':self.extract_automail}

    def create_dict_for_extraction(self):
        if (self.email_uname.create_names_dict() and
                self.email_yob.create_yob_dict() and
                self.email_address.create_address_dict()):
            print('All dictionary data collected!')
            self.dict_built = True
    
    def extract_username(self, df, key_col='email', input_col = 'email_name', merge_col = 'key'):
        # Extract username.
        df_name = self.email_uname.get_username(df.copy(), key_col, input_col, merge_col)
        return df_name
    
    def extract_yob(self, df, key_col='email', input_col = 'email_name', merge_col = 'key'):
        # Extract year of birth.
        df_yob = self.email_yob.get_yearofbirth(df.copy(), key_col, input_col, merge_col)
        return df_yob
    
    def extract_phone(self, df, key_col='email', input_col = 'email_name', merge_col = 'key'):
        # Extract phone.
        df_phone = self.email_phone.get_phone_number(df.copy(), key_col, input_col, merge_col)
        return df_phone
    
    def extract_address(self, df, key_col='email', input_col = 'email_name', merge_col = 'key'):
        # Extract location.
        df_address = self.email_address.get_address(df.copy(), key_col, input_col, merge_col)
        return df_address
    
    def extract_gender(self, df, key_col='email', input_col = 'email_name', merge_col = 'key', name_col='username'):
        # Extract location.
        if not 'username' in df.columns:
            raise ValueError('Extracting Username first!')
        df_filter_uname = df[df[name_col].notna()].reset_index(drop=True)
        df_gender_pred = self.email_gender.get_gender(df_filter_uname.copy(), key_col, input_col, merge_col, name_col)
        map_gender = pd.Series(df_gender_pred['gender'].values, index=df_gender_pred[name_col]).to_dict()
        df_gender = df[[merge_col, key_col, input_col, name_col]]
        df_gender['gender'] = df_gender[name_col].map(map_gender)
        return df_gender[[merge_col, 'gender']]
    
    def extract_group(self, df, key_col='email', input_col = 'email_name', merge_col = 'key'):
        # Extract group.
        df_group = df
        df['email_domain'] = df['email'].str.split('@').str[1]
        df_group_dict = pd.read_parquet(self.path_list.group_dict)
        map_group = pd.Series(df_group_dict['EmailGroup'].values, index=df_group_dict['Domain']).to_dict()
        df_group['email_group'] = df_group['email_domain'].map(map_group)
        return df_group[[merge_col, key_col, input_col, 'email_group']]
    
    def extract_automail(self, df, key_col='email', input_col = 'email_name', merge_col = 'key'):
        # Extract autoemail.
        df_autoemail = df
        is_automail = df['email'].astype(str).str.contains('_autoemail')
        df_autoemail['is_autoemail'] = False
        df_autoemail.loc[is_automail, 'is_autoemail'] = True 
        return df_autoemail
    
    def extract_information(self, df, information_list, key_col='email', input_col = 'email_name', merge_col='key'):
        """Uses built VN location dict to extract information in list."""
        # Returns a df.
        df_info_full = df.copy()
        for info in information_list:
            df_info = self.info_list[info](df.copy(), key_col, input_col, merge_col)
            df_info_full = pd.merge(df_info_full.drop(columns=[key_col, input_col]), df_info, on=[merge_col], how='left')
        return df_info_full