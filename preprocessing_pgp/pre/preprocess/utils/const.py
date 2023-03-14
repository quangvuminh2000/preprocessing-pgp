
import os
import subprocess

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
UTILS_PATH =  f'{ROOT_PATH}/utils'
CENTRALIZE_PATH = f'{ROOT_PATH}/centralize'
PREPROCESS_PATH = f'{ROOT_PATH}/preprocess/enhance'
PRODUCT_PATH = f'{ROOT_PATH}/utils/product'

# * DICT UTILS PATH
DICT_PHONE_UTILS_PATH = f'{UTILS_PATH}/valid_phone_latest.parquet'
DICT_EMAIL_UTILS_PATH = f'{UTILS_PATH}/valid_email_latest.parquet'
DICT_NAME_UTILS_PATH = f'{UTILS_PATH}/dict_name_latest.parquet'

# * DICT PRODUCT PATH
DICT_PHONE_PRODUCT_PATH = f'{PRODUCT_PATH}/valid_phone_latest.parquet'
DICT_EMAIL_PRODUCT_PATH = f'{PRODUCT_PATH}/valid_email_latest.parquet'
DICT_NAME_PRODUCT_PATH = f'{PRODUCT_PATH}/dict_name_latest.parquet'

# * RAW DWH PATH
FO_VNE_PATH = '/data/fpt/ftel/cads/dep_solution/user/trinhlk2/core_profile/build_dict/fo_vne_raw.parquet'
FRT_CREDIT_FE_CREDIT_PATH = '/data/fpt/ftel/cads/dep_solution/user/trinhlk2/core_profile/build_dict/frt_credit_fe_credit_raw.parquet'
FRT_CREDIT_HOME_CREDIT_PATH = '/data/fpt/ftel/cads/dep_solution/user/trinhlk2/core_profile/build_dict/frt_credit_home_credit_raw.parquet'
FRT_CREDIT_MIRAE_ONL_PATH = '/data/fpt/ftel/cads/dep_solution/user/trinhlk2/core_profile/build_dict/frt_credit_mirae_onl_raw.parquet'
FRT_CREDIT_MIRAE_OFF_PATH = '/data/fpt/ftel/cads/dep_solution/user/trinhlk2/core_profile/build_dict/frt_credit_mirae_off_raw.parquet'
FRT_CREDIT_FSHOP_FORM_PATH = '/data/fpt/ftel/cads/dep_solution/sa/cdp/devices/raw/fshop_registration_form_installment.parquet/d=2023-02-26'
FRT_CREDIT_FSHOP_CUSTOMER_PATH = '/data/fpt/ftel/cads/dep_solution/sa/cdp/devices/raw/fshop_customer_installment.parquet/d=2023-02-26'
FRT_FSHOP_PATH = '/data/fpt/ftel/cads/dep_solution/user/trinhlk2/core_profile/build_dict/frt_fshop_raw.parquet'
FRT_LONGCHAU_PATH = '/data/fpt/ftel/cads/dep_solution/user/trinhlk2/core_profile/build_dict/frt_longchau_raw.parquet'
FSOFT_VIO_PATH = '/data/fpt/ftel/cads/dep_solution/user/trinhlk2/core_profile/build_dict/fsoft_vio_raw.parquet'
FTEL_FPLAY_PATH = '/data/fpt/ftel/cads/dep_solution/user/trinhlk2/core_profile/build_dict/ftel_fplay_raw.parquet'
FTEL_INTERNET_DEMO_PATH = '/data/fpt/ftel/cads/dep_solution/user/trinhlk2/core_profile/build_dict/ftel_internet_demo_raw.parquet'
FTEL_INTERNET_MULTI_PATH = '/data/fpt/ftel/cads/dep_solution/user/trinhlk2/core_profile/build_dict/ftel_internet_multi_raw.parquet'
SENDO_SENDO_PATH = '/data/fpt/ftel/cads/dep_solution/user/trinhlk2/core_profile/build_dict/sendo_sendo_raw.parquet'

PATH_DICT = {
    'fo_vne': FO_VNE_PATH,
    'frt_credit_fe_credit': FRT_CREDIT_FE_CREDIT_PATH,
    'frt_credit_home_credit': FRT_CREDIT_HOME_CREDIT_PATH,
    'frt_credit_mirae_onl': FRT_CREDIT_MIRAE_ONL_PATH,
    'frt_credit_mirae_off': FRT_CREDIT_MIRAE_OFF_PATH,
    'frt_credit_fshop_form': FRT_CREDIT_FSHOP_FORM_PATH,
    'frt_credit_fshop_customer': FRT_CREDIT_FSHOP_CUSTOMER_PATH,
    'frt_fshop': FRT_FSHOP_PATH,
    'frt_longchau': FRT_LONGCHAU_PATH,
    'fsoft_vio': FSOFT_VIO_PATH,
    'ftel_fplay': FTEL_FPLAY_PATH,
    'ftel_internet_demo': FTEL_INTERNET_DEMO_PATH,
    'ftel_internet_multi': FTEL_INTERNET_MULTI_PATH,
    'sendo_sendo': SENDO_SENDO_PATH
}



