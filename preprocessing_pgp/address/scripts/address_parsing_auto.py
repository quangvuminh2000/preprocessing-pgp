print('Import python libraries!')
import numpy as np
from datetime import datetime
import os
from os import path
import unidecode
import argparse
import subprocess
from pyarrow import fs
import pyarrow.parquet as pq

import pandas as pd
from glob import glob

import multiprocessing
from multiprocessing import Pool
from functools import partial


from datetime import timedelta
from dateutil.relativedelta import relativedelta

os.environ['HADOOP_CONF_DIR'] = "/etc/hadoop/conf/"
os.environ['JAVA_HOME'] = "/usr/jdk64/jdk1.8.0_112"
os.environ['HADOOP_HOME'] = "/usr/hdp/3.1.0.0-78/hadoop"
os.environ['ARROW_LIBHDFS_DIR'] = "/usr/hdp/3.1.0.0-78/usr/lib/"
os.environ['CLASSPATH'] = subprocess.check_output("$HADOOP_HOME/bin/hadoop classpath --glob", shell=True).decode('utf-8')
hdfs = fs.HadoopFileSystem(host="hdfs://hdfs-cluster.datalake.bigdata.local", port=8020)

print('Import address component!')

from extract_address_component import LocationDict
from extract_address_component import Level1toLevel3Parsing
from extract_address_component import Level4Parsing

print('Import.. Done!')


raw_input_address = '/bigdata/fdp/cdp/profile/rawdata/*.parquet'
outf = '/bigdata/fdp/user/khanhhb3/rundeck/address/posdata'

default_start_date = '2021-12-04'
parser = argparse.ArgumentParser()
parser.add_argument('--last_update', type=str, default = 'latest', help='last update date')
parser.add_argument('--start_date', type=str, default = 'latest', help='start date of email')
opt = parser.parse_args()
# last_update varies according to last update check
last_update = opt.last_update
# start date only has 2 values ['2021-10-03', 'latest']
start_date = opt.start_date


# latest_address_dict = '/bigdata/fdp/user/khanhhb3/rundeck/address/posdata/address_dict_%s.parquet'%start_date
# latest_fshop_address_dict = '/bigdata/fdp/user/khanhhb3/rundeck/address/posdata/fshop_address_%s.parquet'%start_date
# latest_longchau_address_dict = '/bigdata/fdp/user/khanhhb3/rundeck/address/posdata/longchau_address_%s.parquet'%start_date

latest_address_dict = '/data/fpt/ftel/cads/dep_solution/sa/cdp/data/address_dict_%s.parquet'%start_date
latest_fshop_address_dict = '/data/fpt/ftel/cads/dep_solution/sa/cdp/data/fshop_address_%s.parquet'%start_date
latest_longchau_address_dict = '/data/fpt/ftel/cads/dep_solution/sa/cdp/data/longchau_address_%s.parquet'%start_date

outf_hdfs = '/data/fpt/ftel/cads/dep_solution/sa/cdp/data'

print('Path configuration.. Done!')

def is_directory_hdfs(data_dir):
    hdfs_fileinfo = None
    hdfs_fileinfo = hdfs.get_file_info(data_dir)
    return hdfs_fileinfo.type.name in ['File', 'Directory']

def get_latest_address_result(last_update_param):
    df_latest_address = None
    for address_dir in [latest_fshop_address_dict, latest_longchau_address_dict]:
        if not is_directory_hdfs(address_dir):
            raise ValueError("Cannot load last dict")
        df_address = pd.read_parquet(address_dir,
                                     columns=['Address', 'Update'],
                                    filesystem=hdfs)\
                        .drop_duplicates().reset_index(drop=True)
        df_latest_address = pd.concat([df_latest_address, df_address])
    df_latest_address = df_latest_address.rename(columns={'Address':'address', 'Update':'update'})
    df_latest_address = df_latest_address.drop_duplicates().reset_index(drop=True)
    
    if not 'update' in df_latest_address.columns:
        df_latest_address['update'] = default_start_date
        
    if last_update_param == 'latest':
        last_update_param = pd.to_datetime(df_latest_address[df_latest_address['update'].notna()]['update']).sort_values().tail(1).dt.strftime('%Y-%m-%d').item()
    print('Updating from %s'%last_update_param)
    df_latest_address_return = df_latest_address[pd.to_datetime(df_latest_address['update'])<=pd.to_datetime(last_update_param)].reset_index(drop=True)
    
    return df_latest_address_return

def get_hdfs_profile_raw(data_datetime,
                        hdfs_data_dir,
                        data_available_ndays_thr=30):
    ndays_prev = 0
    data_available = False
    while (data_available is False) and (ndays_prev<=data_available_ndays_thr):
        dateobs_str = (data_datetime-relativedelta(days=ndays_prev)).strftime('%Y-%m-%d')
        hdfs_data_file = os.path.join(hdfs_data_dir, 'd=%s'%dateobs_str)
        data_available = is_directory_hdfs(hdfs_data_file)
        ndays_prev+=1
    if data_available is False:
        return None
    print(f'Reading latest profile data: {hdfs_data_file}')
    profile_data_hdfs = pd.read_parquet(hdfs_data_file, filesystem=hdfs)
    return profile_data_hdfs


def get_new_address(df_latest, dateobs, ndays_available_thr):
#     df_prof_fin = None
    print('Only FShop and Long Chau Address')
#     profile_file = ['/bigdata/fdp/cdp/profile/rawdata/profile_fshop.parquet',
#                    '/bigdata/fdp/cdp/profile/rawdata/profile_longchau.parquet']
    
#     for prof_dir in profile_file:
#         df_read = pd.read_parquet(prof_dir)
#         if 'Address' not in df_read.columns:
#             continue
#         else:
#             df_prof = df_read[['Address']].drop_duplicates()
#         df_prof = df_prof.rename(columns={'Address':'address'})
#         df_prof_fin = pd.concat([df_prof_fin, df_prof]).drop_duplicates()
        
    profile_hdfs_file = ['/data/fpt/ftel/cads/dep_solution/sa/dev/raw/fshop.parquet',
           '/data/fpt/ftel/cads/dep_solution/sa/dev/raw/longchau.parquet']
    df_prof_fin_hdfs = None

    for prof_hdfs_dir in profile_hdfs_file:
        try:
            df_read_hdfs = get_hdfs_profile_raw(dateobs, prof_hdfs_dir, ndays_available_thr)
            if df_read_hdfs is None:
                return None
            
            if 'address' not in df_read_hdfs.columns:
                continue
            else:
                df_prof_hdfs = df_read_hdfs[['address']].drop_duplicates()
            df_prof_fin_hdfs = pd.concat([df_prof_fin_hdfs, df_prof_hdfs]).drop_duplicates()
        except Exception as e:
            print('Reading %s failed!'%prof_hdfs_dir)
            return None

    df_prof_fin_hdfs = df_prof_fin_hdfs[~df_prof_fin_hdfs['address'].isin(df_latest['address'])].drop_duplicates()
    
        
    return df_prof_fin_hdfs.reset_index(drop=True)

def parallelize_with_kwargs(data_frame, func, n_cores=8, **kwargs):
    df_split = np.array_split(data_frame, n_cores)
    
    pool = Pool(n_cores)
    
    try:
        dfOutput = pd.concat(pool.map(partial(func, **kwargs), df_split))
    except Exception as e:
        raise e
    finally:
        pool.close()
        
    return dfOutput.copy()

def get_lv1_to_lv3(df, ref_dict, loc_parse_module):
    for lv_col in ['lv1', 'lv2', 'lv3']:
        df[lv_col] = np.nan
    df_input = df
    df_input['lvs_parsed'] = np.nan
    modified_address = []
    for required_lvs in [3, 2, 1]:
        form_parse_list = loc_parse_module.get_diff_form_vnloc(required_lvs)
        
        for order_ in form_parse_list:
            df_parsed = df_input[(df_input['lv1'].isna()) & (df_input['lv2'].isna()) & (df_input['lv3'].isna())].copy()
            out_col1 = 'result_lv%d'%required_lvs
            out_col2 = 'order_lv%d'%required_lvs
            df_parsed[out_col1] = np.nan
            df_parsed[out_col2] = np.nan
            if df_parsed.shape[0] > 100000:
                num_sub_df = int(df_parsed.shape[0])//100000 + 1
            else:
                num_sub_df = 1
                
            df_parsed_split = np.array_split(df_parsed, num_sub_df)
            df_out = None
            for df_address_sub in df_parsed_split:
                df_sub_out = parallelize_with_kwargs(df_address_sub.copy(), loc_parse_module.get_full_level, 20, order_search = order_,
                                                 num_level_required=required_lvs, ref_dict=ref_dict.copy(), output_col1=out_col1,
                                                 output_col2= out_col2, input_col='address_clean')
                df_out = pd.concat([df_out, df_sub_out])
            df_out = df_out.reset_index(drop=True)
            df_input = loc_parse_module.mapping_results(df_input.copy(), df_out, input_col=out_col1, order_decode=out_col2)
            parsed_out_method = (df_out[out_col1].notna())
            modified_address += df_out[parsed_out_method]['address'].unique().tolist()
        df_input.loc[(df_input['address'].isin(modified_address)) & (df_input['lvs_parsed'].isna()), 'lvs_parsed'] = required_lvs
        
    ref_dict_NAccent = ref_dict.copy()
    for col in ['lv1', 'lv2', 'lv3']:
        ref_dict_NAccent[col] = ref_dict_NAccent[col].apply(unidecode.unidecode).str.strip()
    df_outlier = loc_parse_module.handle_outlier_case(df_input.copy(), 'lvs_parsed', ref_dict_NAccent.copy())
    for lv_col in ['lv1', 'lv2', 'lv3']:
        outlier_address_dict = pd.Series(df_outlier[lv_col].values, index=df_outlier['address']).to_dict()
        df_input.loc[df_input['address'].isin(df_outlier['address'].unique().tolist()), lv_col] =\
        df_input.loc[df_input['address'].isin(df_outlier['address'].unique().tolist()),'address'].map(outlier_address_dict)
    return df_input

def get_lv4(df, lv_col, location_dict_data):
    if lv_col not in df.columns:
        df[lv_col] = np.nan
    notnan_data = (df['lv1'].notna()) | (df['lv2'].notna()) |(df['lv3'].notna())
    df_address = df[notnan_data]
    
    my_loc_lv4 = Level4Parsing(location_dict_data.copy())
    df_cleanadd = my_loc_lv4.clean_data(df_address.copy(), input_col = 'address', clean_col = 'address_clean')
    if df_cleanadd[df_cleanadd['lv2'].notna()].shape[0] == 0:
        return df_cleanadd
    df_cleanadd['lv1'] = df_cleanadd['lv1'].str.replace('Tinh |Thanh pho |Thanh Pho ','',regex=True)
    df_cleanadd['address_clean'] = df_cleanadd['address_clean'].str.title()
    for fill_col in ['lv1', 'lv2', 'lv3']:
        df_cleanadd[fill_col] = df_cleanadd[fill_col].fillna('')
    df_cleanadd.loc[df_cleanadd['address_clean'].notna(), lv_col] =\
    df_cleanadd.loc[df_cleanadd['address_clean'].notna()].apply(lambda x: my_loc_lv4.clip_street_address(x.address_clean, x.lv3, x.lv2, x.lv1), axis=1)
    
    df_lv4_postproc = my_loc_lv4.post_process(df_cleanadd.copy(), lv_col)
    df_lv4_postproc['length_%s'%lv_col] = df_lv4_postproc[lv_col].str.len()
    df_lv4_output = df_lv4_postproc[df_lv4_postproc['length_%s'%lv_col]>=3]
    
    map_lv4 = pd.Series(df_lv4_output.lv4.values, index=df_lv4_output.address).to_dict()
    df.loc[df['address'].isin(df_lv4_output['address'].unique().tolist()), lv_col] =\
    df.loc[df['address'].isin(df_lv4_output['address'].unique().tolist()),'address'].map(map_lv4)
    
    return df.drop_duplicates().reset_index(drop=True)


def post_process_full_address(df):
    RETURN_COLS = ['address', 'norm_address', 'street', 'ward', 'district', 'city']
    df = df.rename(columns={'lv1':'city', 'lv2':'district', 'lv3':'ward',
                           'lv4':'street'})
    
    df['norm_address'] = np.nan
    for col in ['street', 'ward', 'district', 'city']:
        if df[col].notna().sum() == 0:
            continue
        df[col] = df[col].str.strip()
        df.loc[df[col].str.len() <= 1, col] = np.nan
        mask = df[col].notna()
        firstfill = df['norm_address'].isna()
        df.loc[mask & firstfill, 'norm_address'] = df.loc[mask & firstfill, 'norm_address'].fillna('') + df.loc[mask & firstfill, col]
        df.loc[mask & (~firstfill), 'norm_address'] = df.loc[mask & (~firstfill), 'norm_address'].fillna('') +', ' + df.loc[mask & (~firstfill), col] 
    return df[RETURN_COLS].reset_index(drop=True)

def map_result(address_dict, cttv_str, update_datetime, ndays_available_thr):
    global latest_fshop_address_dict
    global latest_longchau_address_dict
    
    OUT_COLS = ['CardCode', 'Address', 'NormAddress', 'Street', 'Ward', 'District', 'City', 'Update']
    address_dict = address_dict.rename(columns={'address':'Address', 'norm_address':'NormAddress', 'street':'Street', 'ward':'Ward',
                       'district':'District', 'city':'City', 'update':'Update'})
    
    if cttv_str == 'LongChau':
        longchau_data_dir = '/data/fpt/ftel/cads/dep_solution/sa/dev/raw/longchau.parquet'
        df = get_hdfs_profile_raw(update_datetime, longchau_data_dir, ndays_available_thr)
        df = df.rename(columns={'cardcode_longchau':'CardCode', 'address':'Address'})
#         df_dict = pd.read_parquet(latest_longchau_address_dict, columns=OUT_COLS)
        df_dict = pd.read_parquet(latest_longchau_address_dict,
                                  columns=OUT_COLS,
                                 filesystem=hdfs)
        
    elif cttv_str == 'FSHOP':
        fshop_data_dir = '/data/fpt/ftel/cads/dep_solution/sa/dev/raw/fshop.parquet'
        df = get_hdfs_profile_raw(update_datetime, fshop_data_dir, ndays_available_thr)
        df = df.rename(columns={'cardcode_fshop':'CardCode', 'address':'Address'})
#         df_dict = pd.read_parquet(latest_fshop_address_dict, columns=OUT_COLS)
        df_dict = pd.read_parquet(latest_fshop_address_dict,
                                  columns=OUT_COLS,
                                 filesystem=hdfs)
    
    df_address_update = df[~df['Address'].isin(df_dict['Address'])].drop_duplicates().reset_index(drop=True)
    for map_col in ['NormAddress', 'Street', 'Ward', 'District', 'City', 'Update']:
        map_dict = pd.Series(address_dict[map_col].values, index=address_dict['Address']).to_dict()
        df_address_update[map_col] = df_address_update['Address'].map(map_dict)
        
    print('%s: %d new address updated in %s'%(cttv_str, df_address_update.shape[0], datetime.today().strftime('%Y-%m-%d')))    

    df_out = pd.concat([df_dict, df_address_update]).drop_duplicates().reset_index(drop=True)
        
    if cttv_str == 'LongChau':
#         df_out[OUT_COLS].to_parquet('/bigdata/fdp/user/khanhhb3/rundeck/address/posdata/longchau_address_latest.parquet', index=False, compression='gzip')
        df_out[OUT_COLS].to_parquet(os.path.join(outf_hdfs, 'longchau_address_latest.parquet'), 
            filesystem = hdfs,
            index = False,
            compression = 'gzip'
           )
    elif cttv_str == 'FSHOP':
#         df_out[OUT_COLS].to_parquet('/bigdata/fdp/user/khanhhb3/rundeck/address/posdata/fshop_address_latest.parquet', index=False, compression='gzip')
        df_out[OUT_COLS].to_parquet(os.path.join(outf_hdfs, 'fshop_address_latest.parquet'), 
            filesystem = hdfs,
            index = False,
            compression = 'gzip'
           )
    
    return 0

def update_new_address(df_latest_add, df_new_add, dateobs, ndays_available_thr):
          
    df_new_add['update'] = dateobs.strftime('%Y-%m-%d')
      
#     final_cols = df_new_add.columns.to_list()
#     df_out = pd.concat([df_new_add, df_latest_add[final_cols]])
    df_new_add['update'] = pd.to_datetime(df_new_add['update'])
    for cttv in ['LongChau', 'FSHOP']:
        map_result(df_new_add.copy(), cttv, dateobs, ndays_available_thr)
    
if __name__ == "__main__":
    data_datetime = datetime.today()
    print('Day: %s'%(data_datetime.strftime('%Y-%m-%d')))
    raw_address_available_ndays_thr = 30
    
    df_latest_address = get_latest_address_result(last_update)
    df_new_address_raw = get_new_address(df_latest_address.copy(), data_datetime, raw_address_available_ndays_thr)
    if df_new_address_raw is None:
        print('No new address to process!')
    else:
        print('Processing %d new address..'%(df_new_address_raw.shape[0]))

        df_new_address = df_new_address_raw[df_new_address_raw['address'].str.len() >5].reset_index(drop=True)

        my_loc_dict = LocationDict()
        location_df = my_loc_dict.load_loc_dict()

        my_loc_parse = Level1toLevel3Parsing()

        df_clean_address = my_loc_parse.clean_address(df_new_address.copy(), input_col='address')

        print('Extracting level 1 to level 3..')
        df_3lvs = get_lv1_to_lv3(df_clean_address.copy(), location_df.copy(), my_loc_parse)
        print('Extracting level 4...')
        df_full_lv = get_lv4(df_3lvs.copy(), 'lv4', location_df.copy())
        df_final_address = post_process_full_address(df_full_lv.copy())
        update_new_address(df_latest_address.copy(), df_final_address.copy(),
                           data_datetime, raw_address_available_ndays_thr)

    
