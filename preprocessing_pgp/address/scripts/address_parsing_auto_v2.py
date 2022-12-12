from extract_address_component import Level4Parsing
from extract_address_component import Level1toLevel3Parsing
from extract_address_component import LocationDict
from dateutil.relativedelta import relativedelta
from datetime import timedelta
from functools import partial
from multiprocessing import Pool
import multiprocessing
from glob import glob
import pandas as pd
import pyarrow.parquet as pq
from pyarrow import fs
import subprocess
import argparse
import unidecode
from os import path
import os
from datetime import datetime
import numpy as np
print('Import python libraries!')


os.environ['HADOOP_CONF_DIR'] = "/etc/hadoop/conf/"
os.environ['JAVA_HOME'] = "/usr/jdk64/jdk1.8.0_112"
os.environ['HADOOP_HOME'] = "/usr/hdp/3.1.0.0-78/hadoop"
os.environ['ARROW_LIBHDFS_DIR'] = "/usr/hdp/3.1.0.0-78/usr/lib/"
os.environ['CLASSPATH'] = subprocess.check_output(
    "$HADOOP_HOME/bin/hadoop classpath --glob", shell=True).decode('utf-8')
hdfs = fs.HadoopFileSystem(
    host="hdfs://hdfs-cluster.datalake.bigdata.local", port=8020)


print('Import.. Done!')


raw_input_address = '/bigdata/fdp/cdp/profile/rawdata/*.parquet'
outf = '/bigdata/fdp/user/khanhhb3/rundeck/address/posdata'

default_start_date = '2021-12-04'
# last_update varies according to last update check
last_update = 'latest'
# start date only has 2 values ['2021-10-03', 'latest']
start_date = 'latest'


# latest_address_dict = '/bigdata/fdp/user/khanhhb3/rundeck/address/posdata/address_dict_%s.parquet'%start_date
# latest_fshop_address_dict = '/bigdata/fdp/user/khanhhb3/rundeck/address/posdata/fshop_address_%s.parquet'%start_date
# latest_longchau_address_dict = '/bigdata/fdp/user/khanhhb3/rundeck/address/posdata/longchau_address_%s.parquet'%start_date

latest_address_dict = '/data/fpt/ftel/cads/dep_solution/sa/cdp/data/address_dict_%s.parquet' % start_date
latest_fshop_address_dict = '/data/fpt/ftel/cads/dep_solution/sa/cdp/data/fshop_address_%s.parquet' % start_date
latest_longchau_address_dict = '/data/fpt/ftel/cads/dep_solution/sa/cdp/data/longchau_address_%s.parquet' % start_date

outf_hdfs = '/data/fpt/ftel/cads/dep_solution/sa/cdp/data'

address_home_dir_dict = {'FSHOP': '/data/fpt/ftel/cads/dep_solution/sa/dev/raw/fshop.parquet',
                         'LONGCHAU': '/data/fpt/ftel/cads/dep_solution/sa/dev/raw/longchau.parquet'}

process_address_dir_dict = {'FSHOP': latest_fshop_address_dict,
                            'LONGCHAU': latest_longchau_address_dict}

idcol_dict = {'FSHOP': 'cardcode_fshop',
              'LONGCHAU': 'cardcode_longchau'}

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
                                     columns=['CardCode', 'Address', 'Update'],
                                     filesystem=hdfs)\
            .drop_duplicates().reset_index(drop=True)
        df_latest_address = pd.concat([df_latest_address, df_address])
    df_latest_address = df_latest_address.drop_duplicates().reset_index(drop=True)

    if 'Update' not in df_latest_address.columns:
        df_latest_address['Update'] = default_start_date

    if last_update_param == 'latest':
        last_update_param = pd.to_datetime(df_latest_address[df_latest_address['Update'].notna(
        )]['Update']).sort_values().tail(1).dt.strftime('%Y-%m-%d').item()
    print('Updating from %s' % last_update_param)
    df_latest_address_return = df_latest_address[pd.to_datetime(
        df_latest_address['Update']) <= pd.to_datetime(last_update_param)].reset_index(drop=True)

    return df_latest_address_return


def get_hdfs_profile_raw(data_datetime,
                         hdfs_data_dir,
                         raw_address_col='address',
                         new_address_col='Address',
                         data_available_ndays_thr=30):
    ndays_prev = 0
    data_available = False
    while (data_available is False) and (ndays_prev <= data_available_ndays_thr):
        dateobs_str = (data_datetime-relativedelta(days=ndays_prev)
                       ).strftime('%Y-%m-%d')
        hdfs_data_file = os.path.join(hdfs_data_dir, 'd=%s' % dateobs_str)
        data_available = is_directory_hdfs(hdfs_data_file)
        ndays_prev += 1
    if data_available is False:
        return None
    print(f'Reading latest profile data: {hdfs_data_file}')
    profile_data_hdfs = pd.read_parquet(hdfs_data_file, filesystem=hdfs)
    profile_data_hdfs = profile_data_hdfs.rename(
        columns={raw_address_col: new_address_col})
    return profile_data_hdfs


def get_new_address(df_latest, dateobs, address_list_dict,
                    proc_address_list_dict, raw_idcol_dict,
                    idcol_new, address_raw_col,
                    new_address_col, ndays_available_thr):
    #     df_prof_fin = None
    print('Only FShop and Long Chau Address')
    address_name_list = address_list_dict.keys()
    new_address_total_hdfs = None

    for address_name in address_name_list:
        prof_hdfs_dir = address_list_dict[address_name]
        process_address_dir = proc_address_list_dict[address_name]
        idcol_raw = raw_idcol_dict[address_name]
        try:
            df_read_hdfs = get_hdfs_profile_raw(dateobs, prof_hdfs_dir,
                                                address_raw_col, new_address_col,
                                                ndays_available_thr)
            df_read_hdfs = df_read_hdfs.rename(columns={idcol_raw: idcol_new})
            if df_read_hdfs is None:
                return None
            latest_address_df = pd.read_parquet(process_address_dir,
                                                columns=[idcol_new],
                                                filesystem=hdfs)[idcol_new].unique().tolist()
            new_address_df = df_read_hdfs[~df_read_hdfs[idcol_new].isin(
                latest_address_df)].reset_index(drop=True)

            if new_address_col not in new_address_df.columns:
                print(
                    f'Address column {new_address_col} not found in {address_name} raw data')
                continue

            else:
                new_address_hdfs = new_address_df[[
                    idcol_new, new_address_col]].drop_duplicates()
            new_address_total_hdfs = pd.concat(
                [new_address_total_hdfs, new_address_hdfs]).drop_duplicates()
        except Exception as e:
            print(str(e))
            print('Reading %s failed!' % prof_hdfs_dir)
            return None

    return new_address_total_hdfs.reset_index(drop=True)


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


def get_lv1_to_lv3(df, ref_dict, loc_parse_module,
                   input_col_str, raw_col_str):
    for lv_col in ['lv1', 'lv2', 'lv3']:
        df[lv_col] = np.nan
    df_input = df
    df_input['lvs_parsed'] = np.nan
    modified_address = []
    for required_lvs in [3, 2, 1]:
        form_parse_list = loc_parse_module.get_diff_form_vnloc(required_lvs)

        for order_ in form_parse_list:
            df_parsed = df_input[(df_input['lv1'].isna()) & (
                df_input['lv2'].isna()) & (df_input['lv3'].isna())].copy()
            out_col1 = f'result_lv{required_lvs}'
            out_col2 = f'order_lv{required_lvs}'
            df_parsed[out_col1] = np.nan
            df_parsed[out_col2] = np.nan
            if df_parsed.shape[0] > 100000:
                num_sub_df = int(df_parsed.shape[0])//100000 + 1
            else:
                num_sub_df = 1

            df_parsed_split = np.array_split(df_parsed, num_sub_df)
            df_out = None
            for df_address_sub in df_parsed_split:
                df_sub_out = parallelize_with_kwargs(df_address_sub.copy(), loc_parse_module.get_full_level, 20, order_search=order_,
                                                     num_level_required=required_lvs, ref_dict=ref_dict.copy(), output_col1=out_col1,
                                                     output_col2=out_col2, input_col=input_col_str)
                df_out = pd.concat([df_out, df_sub_out])
            df_out = df_out.reset_index(drop=True)
            df_input = loc_parse_module.mapping_results(df_input.copy(), df_out, input_col=out_col1, order_decode=out_col2,
                                                        raw_col_str=raw_col_str)
            parsed_out_method = (df_out[out_col1].notna())
            modified_address += df_out[parsed_out_method][raw_col_str].unique().tolist()
        df_input.loc[(df_input[raw_col_str].isin(modified_address)) & (
            df_input['lvs_parsed'].isna()), 'lvs_parsed'] = required_lvs

    ref_dict_NAccent = ref_dict.copy()
    for col in ['lv1', 'lv2', 'lv3']:
        ref_dict_NAccent[col] = ref_dict_NAccent[col].apply(
            unidecode.unidecode).str.strip()
    df_outlier = loc_parse_module.handle_outlier_case(
        df_input.copy(), 'lvs_parsed', ref_dict_NAccent.copy())
    for lv_col in ['lv1', 'lv2', 'lv3']:
        outlier_address_dict = pd.Series(
            df_outlier[lv_col].values, index=df_outlier[raw_col_str]).to_dict()
        df_input.loc[df_input[raw_col_str].isin(df_outlier[raw_col_str].unique().tolist()), lv_col] =\
            df_input.loc[df_input[raw_col_str].isin(
                df_outlier[raw_col_str].unique().tolist()), raw_col_str].map(outlier_address_dict)
    return df_input


def get_lv4(df, lv_col, location_dict_data, raw_address_col_str, clean_address_col_str):
    if lv_col not in df.columns:
        df[lv_col] = np.nan
    notnan_data = (df['lv1'].notna()) | (
        df['lv2'].notna()) | (df['lv3'].notna())
    df_address = df[notnan_data]

    my_loc_lv4 = Level4Parsing(location_dict_data.copy())
    df_cleanadd = my_loc_lv4.clean_data(df_address.copy(
    ), input_col=raw_address_col_str, clean_col=clean_address_col_str)
    if df_cleanadd[df_cleanadd['lv2'].notna()].shape[0] == 0:
        return df_cleanadd
    df_cleanadd['lv1'] = df_cleanadd['lv1'].str.replace(
        'Tinh |Thanh pho |Thanh Pho ', '', regex=True)
    df_cleanadd[clean_address_col_str] = df_cleanadd[clean_address_col_str].str.title()
    for fill_col in ['lv1', 'lv2', 'lv3']:
        df_cleanadd[fill_col] = df_cleanadd[fill_col].fillna('')
    df_cleanadd.loc[df_cleanadd[clean_address_col_str].notna(), lv_col] =\
        df_cleanadd.loc[df_cleanadd[clean_address_col_str].notna()].apply(
            lambda x: my_loc_lv4.clip_street_address(x[clean_address_col_str], x.lv3, x.lv2, x.lv1), axis=1)

    df_lv4_postproc = my_loc_lv4.post_process(df_cleanadd.copy(), lv_col)
    df_lv4_postproc['length_%s' % lv_col] = df_lv4_postproc[lv_col].str.len()
    df_lv4_output = df_lv4_postproc[df_lv4_postproc['length_%s' % lv_col] >= 3]

    map_lv4 = pd.Series(df_lv4_output.lv4.values,
                        index=df_lv4_output[raw_address_col_str]).to_dict()
    df.loc[df[raw_address_col_str].isin(df_lv4_output[raw_address_col_str].unique().tolist()), lv_col] =\
        df.loc[df[raw_address_col_str].isin(df_lv4_output[raw_address_col_str].unique(
        ).tolist()), raw_address_col_str].map(map_lv4)

    return df.drop_duplicates().reset_index(drop=True)


def post_process_full_address(df, **kwargs):
    id_col_str = kwargs['id_col_str']
    address_raw_col_str = kwargs['address_raw_col_str']
    address_norm_col_str = kwargs['address_norm_col_str']
    address_lv1_col_str = kwargs['address_lv1_col_str']
    address_lv2_col_str = kwargs['address_lv2_col_str']
    address_lv3_col_str = kwargs['address_lv3_col_str']
    address_lv4_col_str = kwargs['address_lv4_col_str']

    RETURN_COLS = [id_col_str, address_raw_col_str, address_norm_col_str, address_lv4_col_str,
                   address_lv3_col_str, address_lv2_col_str, address_lv1_col_str]

    df = df.rename(columns={'lv1': address_lv1_col_str, 'lv2': address_lv2_col_str,
                            'lv3': address_lv3_col_str, 'lv4': address_lv4_col_str})

    df[address_norm_col_str] = np.nan
    for col in [address_lv4_col_str, address_lv3_col_str,
                address_lv2_col_str, address_lv1_col_str]:

        if df[col].notna().sum() == 0:
            continue
        df[col] = df[col].str.strip()
        df.loc[df[col].str.len() <= 1, col] = np.nan
        mask = df[col].notna()
        firstfill = df[address_norm_col_str].isna()
        df.loc[mask & firstfill, address_norm_col_str] = df.loc[mask & firstfill,
                                                                address_norm_col_str].fillna('') +\
            df.loc[mask & firstfill, col]
        df.loc[mask & (~firstfill), address_norm_col_str] = df.loc[mask & (~firstfill),
                                                                   address_norm_col_str].fillna('') +\
            ', ' + df.loc[mask & (~firstfill), col]
    return df[RETURN_COLS].reset_index(drop=True)


def map_result(address_dict, cttv_str, update_datetime, ndays_available_thr,
               **kwargs):

    proc_address_list_dict = kwargs['proc_address_list_dict']
    id_col_str = kwargs['id_col_str']
    address_raw_col_str = kwargs['address_raw_col_str']
    address_input_col_str = kwargs['address_input_col_str']
    address_norm_col_str = kwargs['address_norm_col_str']
    update_col_str = kwargs['update_col_str']
    address_lv1_col_str = kwargs['address_lv1_col_str']
    address_lv2_col_str = kwargs['address_lv2_col_str']
    address_lv3_col_str = kwargs['address_lv3_col_str']
    address_lv4_col_str = kwargs['address_lv4_col_str']

    address_fshop_dir = proc_address_list_dict['FSHOP']
    address_longchau_dir = proc_address_list_dict['LONGCHAU']

    OUT_COLS = [id_col_str, address_input_col_str, address_norm_col_str, address_lv4_col_str,
                address_lv3_col_str, address_lv2_col_str, address_lv1_col_str, update_col_str]
#     address_dict = address_dict.rename(columns={'address':'Address', 'norm_address':'NormAddress', 'street':'Street', 'ward':'Ward',
#                        'district':'District', 'city':'City', 'Update':'Update'})

    if cttv_str == 'LONGCHAU':
        longchau_data_dir = '/data/fpt/ftel/cads/dep_solution/sa/dev/raw/longchau.parquet'
        df = get_hdfs_profile_raw(update_datetime, longchau_data_dir, address_raw_col_str,
                                  address_input_col_str, ndays_available_thr)
        df = df.rename(
            columns={'cardcode_longchau': 'CardCode', 'address': 'Address'})
        df_dict = pd.read_parquet(address_longchau_dir,
                                  columns=OUT_COLS,
                                  filesystem=hdfs)

    elif cttv_str == 'FSHOP':
        fshop_data_dir = '/data/fpt/ftel/cads/dep_solution/sa/dev/raw/fshop.parquet'
        df = get_hdfs_profile_raw(update_datetime, fshop_data_dir, address_raw_col_str,
                                  address_input_col_str, ndays_available_thr)
        df = df.rename(
            columns={'cardcode_fshop': 'CardCode', 'address': 'Address'})
        df_dict = pd.read_parquet(address_fshop_dir,
                                  columns=OUT_COLS,
                                  filesystem=hdfs)

    df_address_update = df[~df[id_col_str].isin(
        df_dict[id_col_str])].drop_duplicates().reset_index(drop=True)

    [id_col_str, address_input_col_str, address_norm_col_str, address_lv4_col_str,
     address_lv3_col_str, address_lv2_col_str, address_lv1_col_str, update_col_str]

    for map_col in [address_norm_col_str, address_lv4_col_str, address_lv3_col_str,
                    address_lv2_col_str, address_lv1_col_str, update_col_str]:
        map_dict = pd.Series(
            address_dict[map_col].values, index=address_dict[address_input_col_str]).to_dict()
        df_address_update[map_col] = df_address_update[address_input_col_str].map(
            map_dict)
    df_address_update[update_col_str] = df_address_update[update_col_str].fillna(
        datetime.today())
    print('%s: %d new address updated in %s' % (
        cttv_str, df_address_update.shape[0], datetime.today().strftime('%Y-%m-%d')))

    df_out = pd.concat([df_dict, df_address_update])[
        OUT_COLS].drop_duplicates().reset_index(drop=True)

    return df_out


def update_new_address(df_latest_add, df_new_add, dateobs, ndays_available_thr,
                       cttv, **kwargs):
    proc_address_list_dict = kwargs['proc_address_list_dict']
    id_col_str = kwargs['id_col_str']
    address_raw_col_str = kwargs['address_raw_col_str']
    address_input_col_str = kwargs['address_input_col_str']
    address_norm_col_str = kwargs['address_norm_col_str']
    update_col_str = kwargs['update_col_str']
    address_lv1_col_str = kwargs['address_lv1_col_str']
    address_lv2_col_str = kwargs['address_lv2_col_str']
    address_lv3_col_str = kwargs['address_lv3_col_str']
    address_lv4_col_str = kwargs['address_lv4_col_str']

    df_new_add[update_col_str] = dateobs.strftime('%Y-%m-%d')
    df_new_add[update_col_str] = pd.to_datetime(df_new_add[update_col_str])
    new_address_proc = map_result(df_new_add.copy(), cttv, dateobs, ndays_available_thr,
                                  proc_address_list_dict=proc_address_list_dict,
                                  id_col_str=id_col_str,
                                  address_raw_col_str=address_raw_col_str,
                                  address_input_col_str=address_input_col_str,
                                  address_norm_col_str=address_norm_col_str,
                                  update_col_str=update_col_str,
                                  address_lv1_col_str=address_lv1_col_str,
                                  address_lv2_col_str=address_lv2_col_str,
                                  address_lv3_col_str=address_lv3_col_str,
                                  address_lv4_col_str=address_lv4_col_str)
    return new_address_proc


if __name__ == "__main__":
    data_datetime = datetime.today()
    print('Day: %s' % (data_datetime.strftime('%Y-%m-%d')))
    raw_address_available_ndays_thr = 30

    new_id_col_str = 'CardCode'  # rename id col in raw data

    procdata_id_col_str = 'CardCode'
    raw_address_col_str = 'address'
    address_col_str = 'Address'
    address_clean_col_str = 'AddressClean'

    update_colstr = 'Update'
    address_norm_colstr = 'NormAddress'
    address_lv1_colstr = 'City'
    address_lv2_colstr = 'District'
    address_lv3_colstr = 'Ward'
    address_lv4_colstr = 'Street'

    df_latest_address = get_latest_address_result(last_update)
    df_new_address_raw = get_new_address(df_latest_address.copy(), data_datetime,
                                         address_home_dir_dict, process_address_dir_dict, idcol_dict,
                                         new_id_col_str, raw_address_col_str,
                                         address_col_str, raw_address_available_ndays_thr)

    if df_new_address_raw is None:
        print('No new address to process!')
    else:
        print('Processing %d new address..' % (df_new_address_raw.shape[0]))

        df_new_address = df_new_address_raw[df_new_address_raw[address_col_str].str.len(
        ) > 5].reset_index(drop=True)

        my_loc_dict = LocationDict()
        location_df = my_loc_dict.load_loc_dict()

        my_loc_parse = Level1toLevel3Parsing()

        df_clean_address = my_loc_parse.clean_address(df_new_address.copy(
        ), address_col_str=address_col_str, clean_col_str=address_clean_col_str)

        print('Extracting level 1 to level 3..')
        df_3lvs = get_lv1_to_lv3(df_clean_address.copy(), location_df.copy(), my_loc_parse,
                                 address_clean_col_str, address_col_str)

        print('Extracting level 4...')
        df_full_lv = get_lv4(df_3lvs.copy(), 'lv4', location_df.copy(
        ), address_col_str, address_clean_col_str)
        df_final_address = post_process_full_address(df_full_lv.copy(), address_raw_col_str=address_col_str,
                                                     id_col_str=procdata_id_col_str, address_norm_col_str='NormAddress',
                                                     address_lv1_col_str='City', address_lv2_col_str='District',
                                                     address_lv3_col_str='Ward', address_lv4_col_str='Street')

        fshop_latest_address_output = update_new_address(df_latest_address.copy(), df_final_address.copy(),
                                                         data_datetime, raw_address_available_ndays_thr, 'FSHOP',
                                                         proc_address_list_dict=process_address_dir_dict, update_col_str=update_colstr,
                                                         id_col_str=procdata_id_col_str, address_raw_col_str=raw_address_col_str,
                                                         address_input_col_str=address_col_str, address_norm_col_str=address_norm_colstr,
                                                         address_lv1_col_str=address_lv1_colstr, address_lv2_col_str=address_lv2_colstr,
                                                         address_lv3_col_str=address_lv3_colstr, address_lv4_col_str=address_lv4_colstr)
        fshop_proc_address_outf = process_address_dir_dict['FSHOP']
        if fshop_latest_address_output is not None:
            print(f'Saving Fshop process address to {fshop_proc_address_outf}')
            fshop_latest_address_output.to_parquet(fshop_proc_address_outf,
                                                   filesystem=hdfs,
                                                   index=False,
                                                   compression='gzip'
                                                   )

        longchau_latest_address_output = update_new_address(df_latest_address.copy(), df_final_address.copy(),
                                                            data_datetime, raw_address_available_ndays_thr, 'LONGCHAU',
                                                            proc_address_list_dict=process_address_dir_dict, update_col_str=update_colstr,
                                                            id_col_str=procdata_id_col_str, address_raw_col_str=raw_address_col_str,
                                                            address_input_col_str=address_col_str, address_norm_col_str=address_norm_colstr,
                                                            address_lv1_col_str=address_lv1_colstr, address_lv2_col_str=address_lv2_colstr,
                                                            address_lv3_col_str=address_lv3_colstr, address_lv4_col_str=address_lv4_colstr)
        longchau_proc_address_outf = process_address_dir_dict['LONGCHAU']
        if longchau_latest_address_output is not None:
            print(
                f'Saving Long Chau process address to {longchau_proc_address_outf}')
            longchau_latest_address_output.to_parquet(longchau_proc_address_outf,
                                                      filesystem=hdfs,
                                                      index=False,
                                                      compression='gzip'
                                                      )
