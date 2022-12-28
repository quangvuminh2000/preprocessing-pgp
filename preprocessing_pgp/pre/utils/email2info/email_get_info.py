import pandas as pd
import numpy as np
from datetime import datetime

from dateutil.relativedelta import relativedelta
from tqdm.notebook import tqdm

from glob import glob
import os
from os import path
import argparse
import subprocess
from pyarrow import fs
import pyarrow.parquet as pq

import email_extract_2021_08_31
from email_extract_2021_08_31 import EmailCDP

os.environ['HADOOP_CONF_DIR'] = "/etc/hadoop/conf/"
os.environ['JAVA_HOME'] = "/usr/jdk64/jdk1.8.0_112"
os.environ['HADOOP_HOME'] = "/usr/hdp/3.1.0.0-78/hadoop"
os.environ['ARROW_LIBHDFS_DIR'] = "/usr/hdp/3.1.0.0-78/usr/lib/"
os.environ['CLASSPATH'] = subprocess.check_output("$HADOOP_HOME/bin/hadoop classpath --glob", shell=True).decode('utf-8')
hdfs = fs.HadoopFileSystem(host="hdfs://hdfs-cluster.datalake.bigdata.local", port=8020)

datasource_dir = '/bigdata/fdp/cdp/cdp_pages/scripts_hdfs/pre/utils/email2info/config_path.json'
my_email_obj = EmailCDP(datasource_dir)
my_email_obj.create_dict_for_extraction()

def mark_uncertain_username(email_prediction_df):
# column is_certain to mark uncertain username parsed
    private_email_domain_list = ['gmail.com', 'yahoo.com', 'yahoo.com.vn',
                            'icloud.com', 'email.com', 'hotmail.com',
                             'gmai.com', 'outlook.com']
    email_prediction_df['email_domain'] = email_prediction_df['email'].str.split('@').str[1]
    email_prediction_df['private_email'] = email_prediction_df['email_domain'].isin(private_email_domain_list)
    
    email_prediction_df['username_ncharacter'] = email_prediction_df['username'].str.count(' ') +1
    email_prediction_df['username_iscertain'] = ~((email_prediction_df['private_email']==True)
                                    &(email_prediction_df['username_ncharacter']==1))
    email_prediction_df = email_prediction_df.drop(columns=['email_domain', 'private_email',
                                                           'username_ncharacter'])
    return email_prediction_df    

def post_process_email(df_emails, min_len = 5, max_len = 30):
    if 'email_name' not in df_emails.columns:
        email_splits = df_emails['email'].str.split('@', n=1, expand=True)
        df_emails['email_name'] = email_splits[0]

    df_emails['email_len'] = df_emails['email_name'].astype(str).str.len()
    
    # Use regex to check email format.
    regex = '^[0-9]{13,}$'
    df_emails['is_email_valid'] = ~df_emails['email_name'].str.contains(regex, regex=True, na=False)  
    df_emails.loc[(df_emails['email_len'] < min_len) | (df_emails['email_len'] > max_len), 'is_email_valid'] = False
    df_emails = df_emails[df_emails['is_email_valid'] == True]

    return df_emails.drop(columns=['is_email_valid', 'email_len']).drop_duplicates().reset_index(drop=True)


def pipeline_email_to_info(df_in):
    df_in['email'] = df_in['email'].str.lower()
    df_in = df_in.reset_index(drop=True).reset_index()
    df_in = df_in.rename(columns={'index':'key'})
    df_in['email_name'] = df_in['email'].str.split('@').str[0]
    df_info = my_email_obj.extract_information(df_in.copy(), ['username', 'yob', 'phone', 'address', 'group', 'automail'])

    df_info_gender = my_email_obj.extract_information(df_info.copy(), ['gender'])
    df_info_gender = df_info_gender[['key', 'gender']]
    df_out = pd.merge(df_info, df_info_gender, how='left', on=['key'])

    out_cols = df_out.drop(columns=['key']).columns.to_list()

    df_out.insert(0,"email_name",df_out.pop("email_name"))
    df_out.insert(0,"email",df_out.pop("email"))
    df_emails_out = mark_uncertain_username(df_out.copy())

    df_poproc_email = post_process_email(df_emails_out.copy())
    
    df_poproc_email = df_poproc_email.drop(columns=['key']).drop_duplicates()
    
    return df_poproc_email
