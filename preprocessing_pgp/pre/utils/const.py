
import os
import subprocess

import pandas as pd
from pyarrow import fs

os.environ['HADOOP_CONF_DIR'] = "/etc/hadoop/conf/"
os.environ['JAVA_HOME'] = "/usr/jdk64/jdk1.8.0_112"
os.environ['HADOOP_HOME'] = "/usr/hdp/3.1.0.0-78/hadoop"
os.environ['ARROW_LIBHDFS_DIR'] = "/usr/hdp/3.1.0.0-78/usr/lib/"
os.environ['CLASSPATH'] = subprocess.check_output(
    "$HADOOP_HOME/bin/hadoop classpath --glob", shell=True).decode('utf-8')
hdfs = fs.HadoopFileSystem(
    host="hdfs://hdfs-cluster.datalake.bigdata.local", port=8020)


# * NECESSARY PATHS
ROOT_PATH = '/data/fpt/ftel/cads/dep_solution/sa/cdp/core'
UTILS_PATH = f'{ROOT_PATH}/utils'
RAW_PATH = f'{ROOT_PATH}/raw'
UNIFY_PATH = f'{ROOT_PATH}/pre'
PRODUCT_PATH = '/data/fpt/ftel/cads/dep_solution/sa/cdp/data'
VALID_PHONE_PATH = f'{ROOT_PATH}/utils/valid_phone_latest.parquet'
VALID_EMAIL_PATH = f'{ROOT_PATH}/utils/valid_email_latest.parquet'

# * REQUIRED DICTS
# VALID_PHONE_DICT = pd.read_parquet(
#     VALID_PHONE_PATH,
#     filesystem=hdfs
# )

# VALID_EMAIL_DICT = pd.read_parquet(
#     VALID_PHONE_PATH,
#     filesystem=hdfs
# )

# DICT_NAME_LATEST = pd.read_parquet()
