
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
ROOT_PATH = '/data/fpt/ftel/cads/dep_solution/sa/cdp/core_profile'
UTILS_PATH = f'{ROOT_PATH}/utils'
CENTRALIZE_PATH = f'{ROOT_PATH}/centralize'
PREPROCESS_PATH = f'{ROOT_PATH}/preprocess'
PRODUCT_PATH = f'{ROOT_PATH}/utils/product'

# * DICT PATH
DICT_PHONE_PATH = f'{UTILS_PATH}/valid_phone_latest.parquet'
DICT_EMAIL_PATH = f'{UTILS_PATH}/valid_email_latest.parquet'
DICT_NAME_PATH = f'{UTILS_PATH}/dict_name_latest.parquet'

# * CTTV PATH

# * BEST NAME PATH
EMAIL_BEST_NAME_PATH = f'{UTILS_PATH}/name_by_email_latest.parquet'
PHONE_BEST_NAME_PATH = f'{UTILS_PATH}/name_by_phone_latest.parquet'

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
