import numpy as np
import re
import unidecode
import time

import pandas as pd
import itertools
import json

from flashtext import KeywordProcessor

import os
from os import path
import unidecode
import argparse
import subprocess
from pyarrow import fs
import pyarrow.parquet as pq

os.environ['HADOOP_CONF_DIR'] = "/etc/hadoop/conf/"
os.environ['JAVA_HOME'] = "/usr/jdk64/jdk1.8.0_112"
os.environ['HADOOP_HOME'] = "/usr/hdp/3.1.0.0-78/hadoop"
os.environ['ARROW_LIBHDFS_DIR'] = "/usr/hdp/3.1.0.0-78/usr/lib/"
os.environ['CLASSPATH'] = subprocess.check_output("$HADOOP_HOME/bin/hadoop classpath --glob", shell=True).decode('utf-8')
hdfs = fs.HadoopFileSystem(host="hdfs://hdfs-cluster.datalake.bigdata.local", port=8020)


def rreplace(string, substr_remove):
    old_string = string
    
    k_1 = old_string.rfind(substr_remove)
    k_2 = k_1+ len(substr_remove)
    new_string = old_string[:k_1] + old_string[k_2:]
    return new_string

def pad_number(match):
    number = int(match.group(1))
    return format(number, "01d")

def clean_dict_address_wdigit(row):
    text = row
    try:
        text_found = re.search('(Phường|Quận) [0-9]+', text).group(0)
        text = text.replace(text_found, text_found.replace(' ',''))
    except Exception as e:
        text = text
    return text

class LocationDict:
    def __init__(self):
        self.default_loc_dict_dir = '/data/fpt/ftel/cads/dep_solution/sa/cdp/data/location_dict_enrich_address.parquet/d=2022-09-15'
        
    def load_loc_dict(self,
                     loc_dict_dir=None):
        if loc_dict_dir is None:
            loc_dict_dir = self.default_loc_dict_dir
        df_location = None
        print(f'Loading Location Dictionary in {loc_dict_dir}...')
        try:
            df_location = pd.read_parquet(loc_dict_dir,
                                     filesystem=hdfs)
        except Exception as e:
            print(str(e))
        return df_location

class Level1toLevel3Parsing:
    def __init__(self, ):
        self.lv1_method = ['lv1_norm', 'lv1_abbrev', 'lv1_prefix_im', 'lv1_nprefix_im']
        self.lv2_method = ['lv2_norm', 'lv2_abbrev', 'lv2_prefix_im', 'lv2_nprefix_im']
        self.lv3_method = ['lv3_norm', 'lv3_abbrev', 'lv3_prefix_im', 'lv3_nprefix_im']
        
        
    def validate_found_level(self, ref_dict, found_item, validate_item, found_format, validate_format):
        check_df = ref_dict[ref_dict[found_format] == found_item][[found_format, validate_format]].drop_duplicates()
        if check_df[validate_format].shape[0]<1:
            return False
        check_items = check_df[validate_format].to_list()

        return True if validate_item in check_items  else False
    
    
    def prefix_loc_number(self, row):
        text = row
        try:
            text_found = re.search(r'[^A|a]p [0-9]+', text).group(0).strip()
            text = text.replace(text_found, text_found.replace(' ',''))
        except:
            text = text
        try:
            text_found = re.search('q [0-9]+', text).group(0)
            text = text.replace(text_found, text_found.replace(' ',''))
        except:
            text = text
        return text        
        
    def clean_address(self, df, **kwargs):
        input_col = kwargs['address_col_str']
        clean_col = kwargs['clean_col_str']
        
        df[clean_col] = df[input_col].str.replace('\s+', ' ', regex=True).str.lower().apply(unidecode.unidecode)
        abbrev_list = ['tp.', 'tt.', 'q.', 'h.', 'x.', 'p.']
        norm_list = ['tp ', 'tt ', 'q ', 'h ', 'x ', 'p ']
        for case_abbrev, case_norm in zip(abbrev_list, norm_list):
            df[clean_col] = df[clean_col].str.replace(case_abbrev, case_norm, regex=False)
        df[clean_col] = df[clean_col].str.replace('\s+', ' ', regex=True)
        df[clean_col] = df[clean_col].apply(lambda x: re.sub(r"(\d+)",pad_number,x))
        df[clean_col] = df[clean_col].apply(self.prefix_loc_number)
        df[clean_col] = df[clean_col].str.replace('ba ria vung tau', 'ba ria - vung tau', regex=False)
        df[clean_col] = df[clean_col].str.replace('phan rang thap cham', 'phan rang - thap cham', regex=False)
        return df

    def pat_matching(self, query, keyword_processor):
        keywords_found = keyword_processor.extract_keywords(query)
        return keywords_found[-1] if len(keywords_found) >0 else ''
    
    def get_full_address(self, data, **kwargs):
        kw_proc_1 = kwargs['keyword_processor_1']
        level_start_from = kwargs['level_start_from']

        format_1 = kwargs['format_1']
        format_2 = kwargs['format_2']    
        format_3 = kwargs['format_3']

        num_lv_req = kwargs['num_lv_req']

        format_1_norm = format_1[:3]
        format_2_norm = format_2[:3]    
        format_3_norm = format_3[:3]
        ref_dict = kwargs['ref_dict']

        query = data
        is_ok=3
        found_1 = ''
        found_2 = ''
        found_3 = ''
        found_1 = self.pat_matching(query, kw_proc_1)
#         print('found_1',found_1)
        query = rreplace(query, found_1)
        if len(found_1)<2:
            is_ok -=1
            found_1 = ''
        if found_1 == '':
            return '%s, %s, %s'%(found_1, found_2, found_3)
        query = rreplace(query, 'tp ')

        ref_dict_2 = ref_dict[ref_dict[format_1] == found_1]
        kw_proc_2 = self.get_keywordprocess(format_2, ref_dict_2.copy(), level_start_from)

        found_2 = self.pat_matching(query, kw_proc_2)
#         print('found_2',found_2)
        query = rreplace(query, found_2)
        check_found_2 = self.validate_found_level(ref_dict, found_2, found_1, format_2, format_1)
        if check_found_2 is False or len(found_2)<2:
            is_ok -=1
            found_2=''
        if (found_2 == '') and (is_ok < num_lv_req):
            return '%s, %s, %s'%(found_1, found_2, found_3)

        ref_dict_3 = ref_dict[(ref_dict[format_1] == found_1) & (ref_dict[format_2] == found_2)]
        kw_proc_3 = self.get_keywordprocess(format_3, ref_dict_3.copy(), level_start_from)

        for case in ['q ', 'tp ']:
            query = rreplace(query, case)
        found_3 = self.pat_matching(query, kw_proc_3)
#         print('found_3',found_3)
        check_found_3 = self.validate_found_level(ref_dict, found_3, found_1, format_3, format_1)
        if check_found_3 is False or len(found_3)<2:
            is_ok -=1
            found_3=''
        if (found_3 == '') and (is_ok < num_lv_req):
            return '%s, %s, %s'%(found_1, found_2, found_3)
        if is_ok >= num_lv_req:
            if num_lv_req == 3:
                format_dict = ref_dict[(ref_dict[format_1] == found_1) & (ref_dict[format_2] == found_2) & (ref_dict[format_3] == found_3)]
            elif num_lv_req == 2:
                format_dict = ref_dict[(ref_dict[format_1] == found_1) & (ref_dict[format_2] == found_2)]
            elif num_lv_req == 1:
                format_dict = ref_dict[(ref_dict[format_1] == found_1)]

            norm_format_dict = format_dict[format_1_norm].unique()
            if norm_format_dict.shape[0] == 1:
                found_1 = norm_format_dict[0]
            else:
                found_1 = ''
                
            if len(found_2) > 1:
                norm_format_dict = format_dict[format_2_norm].unique()
                lv2_convert_tp = False
                lv2_convert_quan = False
                lv2_convert_thixa = False
                if norm_format_dict.shape[0] == 1 or len(found_1)>1:
                    found_2 = norm_format_dict[0]
                else:
                    found_2 = ''
            else:
                found_2 = ''
                
            if len(found_3) > 1:
                lv3_from_lv1_2 = (format_dict[format_1_norm] == found_1) & (format_dict[format_2_norm] == found_2)
                norm_format_dict_lv3 = format_dict[lv3_from_lv1_2][format_3_norm].unique()

                if norm_format_dict_lv3.shape[0] == 1 or len(found_2)>1:
                    found_3 = norm_format_dict_lv3[0]
                else:
                    found_3 = ''
            else:
                found_3 = ''
        return '%s, %s, %s'%(found_1, found_2, found_3)
    
    def get_pat_dict(self, df_dictionary, lv_method, lv_start_from):
        level_name = lv_method[:3]
        if level_name == lv_start_from:
            df_unique_dict = df_dictionary[[level_name, lv_method]].drop_duplicates()
        else:
            df_unique_dict = df_dictionary[[lv_method]].drop_duplicates()

        pat_finding = df_unique_dict[~df_unique_dict[lv_method].duplicated(keep=False)][lv_method].unique().tolist()
        return pat_finding

    def get_keywordprocess(self, lv_method, ref_dict, lv_start_from):
        pat_finding = self.get_pat_dict(ref_dict.copy(), lv_method, lv_start_from)
        keyword_processor = KeywordProcessor(case_sensitive=True)
        for pattern in pat_finding:
            keyword_processor.add_keyword(pattern)
        return keyword_processor
    
    def get_string_parsed(self, row, **kwargs):
        kw_proc_1 = kwargs['keyword_processor_1']
        level_start_from = kwargs['level_start_from']

        format_1 = kwargs['format_1']
        format_2 = kwargs['format_2'] 
        format_3 = kwargs['format_3']

        input_col = kwargs['input_column']

        ref_dict = kwargs['ref_dict'] 

        num_level_required = kwargs['num_level_required']
        string = row[input_col]
        tmp = self.get_full_address(string, keyword_processor_1=kw_proc_1, format_1=format_1, format_2=format_2, format_3=format_3,
                               level_start_from=level_start_from, ref_dict=ref_dict, num_lv_req=num_level_required)
        num_level_found = 0
        for component in tmp.split(','):
            if len(component.replace(' ','')) > 0:
                num_level_found+=1
        if num_level_found >= num_level_required:
            return tmp
        else:
            return np.nan

    def get_full_level(self, df_data, **kwargs):
        order_search = kwargs['order_search']
        output_col1 = kwargs['output_col1']
        output_col2 = kwargs['output_col2']

        input_col = kwargs['input_col']

        ref_dict = kwargs['ref_dict']
        num_level_required = kwargs['num_level_required']
        df = df_data.copy()
        for order_lv_extract in order_search:
            df = df[df[output_col1].isna()]
            format1 = order_lv_extract[0]
            format2 = order_lv_extract[1]
            format3 = order_lv_extract[2]
            level_start_from = format1[:3]
            kw_processor1 = self.get_keywordprocess(format1, ref_dict.copy(), level_start_from)
            df[output_col1] = df.apply(self.get_string_parsed, keyword_processor_1=kw_processor1, level_start_from=level_start_from,
                                        format_1=format1, format_2=format2, format_3=format3, ref_dict=ref_dict.copy(), input_column = input_col,
                                        num_level_required=num_level_required, axis=1)
            tmp_df = df[df[output_col1].notna()].copy()
            map_address = pd.Series(tmp_df[output_col1].values, index=tmp_df[input_col]).to_dict()
            update_condition = df_data[output_col1].isna()
            df_data.loc[update_condition, output_col1] = df_data.loc[update_condition, input_col].map(map_address)
            update_method = update_condition & (df_data[output_col1].notna())
            df_data.loc[update_method, output_col2] = '%s|%s|%s'%(order_lv_extract[0], order_lv_extract[1], order_lv_extract[2])

        return df_data
    
    def post_process(self, df, **kwargs):
        input_col = kwargs['input_col']
        raw_col_str = kwargs['raw_col_str']
        order_decode = kwargs['order_decode']
        for lv_col in ['lv1', 'lv2', 'lv3']:
            df[lv_col] = np.nan
        
        try:
            if df[input_col].str.count(',').max() == 2:
                print('Address containing 3 levels !')
        except Exception as e:
            print(str(e))
            print('Post processing failed!')
            return df
        print('Decoding results!')
        
        temp_val_df = df[input_col].str.split(',', expand=True)
        temp_order_df = df[order_decode].str.split('|', expand=True)

        for val_col, order_col in zip(temp_val_df.columns, temp_order_df.columns):
            temp_order_df['level_col_name'] = temp_order_df[order_col].str[:3]
            temp_val_df.loc[temp_val_df[val_col].str.len()<2, val_col] = np.nan
    #         print('temp_order_df',temp_order_df)
            for lv_col in ['lv1', 'lv2', 'lv3']:
                df.loc[temp_order_df['level_col_name'] == lv_col, lv_col] = temp_val_df.loc[temp_order_df['level_col_name'] == lv_col, val_col]
        return df[[raw_col_str, 'lv1', 'lv2', 'lv3']]
    
    def mapping_results(self, df, df_parsed, **kwargs):
        raw_col_str = kwargs['raw_col_str']
        input_col = kwargs['input_col']
        order_decode = kwargs['order_decode']
        address_available = df_parsed[df_parsed[input_col].notna()].reset_index(drop=True)
        if address_available is None:
            mapping_df = address_available.copy()
        elif address_available.shape[0] == 0:
            mapping_df = address_available.copy()
        else:
            mapping_df = self.post_process(address_available.copy(), input_col=input_col,
                                           order_decode=order_decode, raw_col_str=raw_col_str)
        
        map_add_lv1 = pd.Series(mapping_df['lv1'].values, index=mapping_df[raw_col_str]).to_dict()
        map_add_lv2 = pd.Series(mapping_df['lv2'].values, index=mapping_df[raw_col_str]).to_dict()
        map_add_lv3 = pd.Series(mapping_df['lv3'].values, index=mapping_df[raw_col_str]).to_dict()

        df.loc[df['lv1'].isna(),'lv1'] = df.loc[df['lv1'].isna(),raw_col_str].map(map_add_lv1)
        df.loc[df['lv2'].isna(),'lv2'] = df.loc[df['lv2'].isna(),raw_col_str].map(map_add_lv2)
        df.loc[df['lv3'].isna(),'lv3'] = df.loc[df['lv3'].isna(),raw_col_str].map(map_add_lv3)
        for col in ['lv1', 'lv2', 'lv3']:
            if df.loc[df[col].notna()].shape[0] == 0:
                continue
            df.loc[df[col].notna(), col] = df.loc[df[col].notna(), col].str.strip().apply(unidecode.unidecode)

        return df
    
    def get_3lvs_dict(self):
        lv1_method = self.lv1_method
        lv2_method = self.lv2_method
        lv3_method = self.lv3_method
        from_lv1 = [lv1_method, lv2_method, lv3_method]
        order_3lvs_lv1 = list(itertools.product(*from_lv1))
        for order in order_3lvs_lv1:
            if 'nprefix' in order[1] and 'nprefix' in order[2]:
                order_3lvs_lv1.remove(order)

        from_lv2 = [lv2_method, lv3_method, lv1_method]
        order_3lvs_lv2 = list(itertools.product(*from_lv2))
        for order in order_3lvs_lv2:
            if 'nprefix' in order[1] and 'nprefix' in order[2]:
                order_3lvs_lv2.remove(order)

        from_lv3 = [lv3_method, lv2_method, lv1_method]
        order_3lvs_lv3 = list(itertools.product(*from_lv3))
        for order in order_3lvs_lv3:
            if 'nprefix' in order[1] and 'nprefix' in order[2]:
                order_3lvs_lv3.remove(order)
        return [order_3lvs_lv1, order_3lvs_lv2, order_3lvs_lv3]
    
    def get_2lvs_dict(self):
        lv1_method = self.lv1_method
        lv2_method = self.lv2_method
        lv3_method = self.lv3_method
        from_lv1_1 = [lv1_method, lv2_method[:3], lv3_method[:1]]
        from_lv1_2 = [lv1_method, lv3_method[:3], lv2_method[:1]]

        from_lv2_1 = [lv2_method, lv1_method[:3], lv3_method[:1]]
        from_lv2_2 = [lv2_method, lv3_method[:3], lv1_method[:1]]

        from_lv3_1 = [lv3_method, lv2_method[:3], lv1_method[:1]]
        from_lv3_2 = [lv3_method, lv1_method[:3], lv2_method[:1]]

        order_2lvs_1 = list(itertools.product(*from_lv1_1)) + list(itertools.product(*from_lv2_1)) + list(itertools.product(*from_lv3_1))
        order_2lvs_2 = list(itertools.product(*from_lv1_2)) + list(itertools.product(*from_lv2_2)) + list(itertools.product(*from_lv3_2))
        for order in order_2lvs_1:
            if 'nprefix' in order[1] and 'nprefix' in order[2]:
                order_2lvs_1.remove(order)
        for order in order_2lvs_2:
            if 'nprefix' in order[1] and 'nprefix' in order[2]:
                order_2lvs_2.remove(order)
        return [order_2lvs_1, order_2lvs_2]
    
    
    def get_1lv_dict(self):
        lv1_method = self.lv1_method
        lv2_method = self.lv2_method
        lv3_method = self.lv3_method
        from_lv1 = [lv1_method[:3], lv3_method[:1], lv2_method[:1]]
        from_lv2 = [lv2_method[:3], lv3_method[:1], lv1_method[:1]]
        from_lv3 = [lv3_method[:1], lv2_method[:1], lv1_method[:1]]

        order_1lv = list(itertools.product(*from_lv1)) + list(itertools.product(*from_lv2)) + list(itertools.product(*from_lv3))
        return [order_1lv]
    
    def set_lv_method(lv, lv_form):
        if lv == 'level1':
            self.lv1_method = lv_form
        elif lv == 'level2':
            self.lv2_method = lv_form
        elif lv == 'level3':
            self.lv3_method = lv_form
    
    
    def get_diff_form_vnloc(self, num_lvs):
        orders_list = []

        if num_lvs == 3:
            orders_list = self.get_3lvs_dict()
        elif num_lvs == 2:
            orders_list = self.get_2lvs_dict()
        elif num_lvs == 1:
            orders_list = self.get_1lv_dict()
        return orders_list
    
    def infer_otherloc_from1lv(self, df, pre_lv, lv, ref_dict):
        df_uniq_rel = ref_dict[[pre_lv, lv]].drop_duplicates()
        df_uniq_rel = df_uniq_rel[~df_uniq_rel[lv].duplicated(keep=False)]

        map_lv = pd.Series(df_uniq_rel[pre_lv].values, index=df_uniq_rel[lv]).to_dict()
        condition = (df[pre_lv].isna()) & (df[lv].notna())
        df.loc[condition, pre_lv] = df.loc[condition, lv].map(map_lv)
        df.loc[condition & (df[pre_lv].notna()), 'reverse'] = '%s-%s'%(lv, pre_lv)
        return df
    
    def infer_otherloc_from2lvs(self, df, pre_lv, lv1str_infer, lv2str_infer, ref_dict):
        lv = '%s_%s'%(lv1str_infer, lv2str_infer)
        df[lv] = df[lv1str_infer].str.strip() + df[lv2str_infer].str.strip()
        ref_dict[lv] = ref_dict[lv1str_infer].str.strip() + ref_dict[lv2str_infer].str.strip()
        df_uniq_rel = ref_dict[[pre_lv, lv]].drop_duplicates()
        df_uniq_rel = df_uniq_rel[~df_uniq_rel[lv].duplicated(keep=False)]
        map_lv = pd.Series(df_uniq_rel[pre_lv].values, index=df_uniq_rel[lv]).to_dict()
        condition = (df[pre_lv].isna()) & (df[lv].notna())
        df.loc[condition, pre_lv] = df.loc[condition, lv].map(map_lv)
        df.loc[condition & (df[pre_lv].notna()), 'reverse'] = '%s-%s'%(lv, pre_lv)
        return df
    
    def handle_outlier_case(self, df, num_lv_parsed_col, ref_dict):
        
        df_outlier = df[df[num_lv_parsed_col]<3].copy()
        df_outlier['modified']=False
        
        df_infer_1_lv = df_outlier.copy()
        df_infer_1_lv['reverse'] = np.nan
        
        infer_from_list = ['lv3', 'lv2']
        infer_to_list = ['lv2', 'lv1']
        for infer_from, infer_to in zip(infer_from_list, infer_to_list):
            df_infer_reversed = self.infer_otherloc_from1lv(df_infer_1_lv.copy(), infer_from, infer_to, ref_dict)
        df_infer_2_lvs = df_infer_reversed[df_infer_reversed['reverse'].isna()]
        
        df_infer = df_infer_2_lvs.copy()
        infer2lvs_frlist_1 = ['lv3', 'lv3']
        infer2lvs_frlist_2 = ['lv2', 'lv1']
        infer2lvs_tolist = ['lv1', 'lv2']
        for infer_from1, infer_from2, infer_to in zip(infer2lvs_frlist_1, infer2lvs_frlist_2, infer2lvs_tolist):
            if df_infer_2_lvs[infer_from1].notna().sum() == 0 or df_infer_2_lvs[infer_from2].notna().sum() == 0:
                continue
            
            df_infer = self.infer_otherloc_from2lvs(df_infer.copy(), infer_to, infer_from1, infer_from2, ref_dict)
        
        
        thuduc_case = ref_dict[ref_dict['lv2'] == 'Thanh pho Thu Duc']['lv3'].unique().tolist()
        thuduc_condition = (df_infer['lv3'].isin(thuduc_case)) & (df_infer['lv1'] == 'Thanh pho Ho Chi Minh')&\
                       (df_infer['lv2'].isna()) & (df_infer['reverse'].isna())
        df_infer.loc[thuduc_condition,'lv2'] = 'Thanh pho Thu Duc'
        df_infer.loc[thuduc_condition,'modified'] = 'True'
        
        return df_infer
    
class Level4Parsing:
    def __init__(self, df_dvhc):
        self.df_location = df_dvhc
        
    def prefix_loc_number(self, row):
        text = row
        try:
            text_found = re.search(r'[^A|a]p [0-9]+', text).group(0).strip()
            text = text.replace(text_found, text_found.replace(' ',''))
        except:
            text = text
        try:
            text_found = re.search('q [0-9]+', text).group(0)
            text = text.replace(text_found, text_found.replace(' ',''))
        except:
            text = text
        return text
    
    def clean_data(self, df, **kwargs):
        input_col = kwargs['input_col']
        clean_col = kwargs['clean_col']

        df[clean_col] = df[input_col].str.replace('\s+', ' ', regex=True)
        df[clean_col] = df[clean_col].apply(unidecode.unidecode).str.lower()
        redundant_space_case = ['P. ', 'p. ', 'H. ', 'h. ', 'X. ', 'x.', 'T. ', 't. ']
        correct_case = ['P.', 'p.', 'H.', 'h.', 'X.', 'x.', 'T.', 't.']
        for case_redundant, case_correct in zip(redundant_space_case, correct_case):
            df[clean_col] = df[clean_col].str.replace(case_redundant, case_correct, regex=False)

        abbrev_case = ['tp.', 'q.', 'p.', ' d.', ' đ ', 'đ. ', 'kp', 'kkt', 'kcn']
        norm_case =   ['thanh pho ', 'quan ', 'phuong ', ' duong ', ' duong ', ' duong ', 'khu pho', 'khu kinh te', 'khu cong nghiep']

        for abbrev, norm in zip(abbrev_case, norm_case):
            df[clean_col] = df[clean_col].str.replace(abbrev, norm, regex=False)
        df[clean_col] = df[clean_col].apply(unidecode.unidecode)
        df[clean_col] = df[clean_col].str.replace('\s+', ' ', regex=True)
        df[clean_col] = df[clean_col].apply(lambda x: re.sub(r"(\d+)",pad_number,x))
        df[clean_col] = df[clean_col].str.replace('/ ',' ', regex=False)
        df[clean_col] = df[clean_col].apply(self.prefix_loc_number)
        df[clean_col] = df[clean_col].str.replace(' ,',',', regex=False)
        df[clean_col] = df[clean_col].str.replace(' .','.', regex=False)
        df[clean_col] = df[clean_col].str.title()
        
        return df

    def clip_street_address(self, address, ward, district, city):
        result = address.title()
        if ward not in [np.nan] and len(ward)>0:
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
        if district not in [np.nan] and len(district)>0:
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
        if city not in [np.nan] and len(city)>0:
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
        return result
    
    def match_spec_char(self, match):
        return match.group(0)[1:]
    
    def post_process(self, df, col2proc):
        df.loc[df[col2proc].notna(),col2proc] = df.loc[df[col2proc].notna(),col2proc].str.replace('[,;!@#%&.]',' ').str.title()

        df.loc[df[col2proc].notna(),col2proc] = df.loc[df[col2proc].notna(),col2proc].str.replace('\s+', ' ', regex=True)
        df.loc[df[col2proc].notna(),col2proc] = df.loc[df[col2proc].notna(),col2proc].str.replace(' - ','-', regex=False)
        for case in ["P;", "T;", "X;",'Dia chi', "N'", ":"]:
            df.loc[df[col2proc].notna(),col2proc] = df.loc[df[col2proc].notna(),col2proc].str.replace(case,"", regex=False)


        df.loc[df[col2proc].notna(),col2proc] = df.loc[df[col2proc].notna(),col2proc].str.replace(r'[^[a-zA-Z0-9| |,|-|.|()|/]]', ' ')
        df.loc[df[col2proc].notna(),col2proc] = df.loc[df[col2proc].notna(),col2proc].str.replace('()', '', regex=False)

        df.loc[df[col2proc].notna(),col2proc] = df.loc[df[col2proc].notna(),col2proc].str.strip()
        df.loc[df[col2proc].notna(),col2proc] = df.loc[df[col2proc].notna(),col2proc].str.replace(' (P|X|Tt|Tx|Q|H|Tp)$','')
        df.loc[df[col2proc].notna(),col2proc] = df.loc[df[col2proc].notna(),col2proc].apply(lambda x: re.sub('(/|-)+[^\d]', self.match_spec_char,x))
        df.loc[df[col2proc].notna(),col2proc] = df.loc[df[col2proc].notna(),col2proc].apply(lambda x: re.sub('[(].*[^)]$','',x))
        df.loc[df[col2proc].notna(),col2proc] = df.loc[df[col2proc].notna(),col2proc].str.replace('(/|-){1,}$','')
        df.loc[df[col2proc].notna(),col2proc] = df.loc[df[col2proc].notna(),col2proc].str.replace('\s+', ' ', regex=True)
        df.loc[df[col2proc].notna(),col2proc] = df.loc[df[col2proc].notna(),col2proc].str.strip()
        return df