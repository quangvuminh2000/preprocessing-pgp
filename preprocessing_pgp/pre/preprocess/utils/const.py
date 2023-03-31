
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
UTILS_PATH = f'{ROOT_PATH}/utils'
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
OLD_DICT_RAW_PATH = '/data/fpt/ftel/cads/dep_solution/sa/rst/raw'
OLD_DICT_CLEAN_PATH = '/data/fpt/ftel/cads/dep_solution/sa/rst/clean'


# ? EMAIL DICT
FO_VNE_PATH = '/data/fpt/fo/dwh/stag_user_profile.parquet' # run_date
FRT_CREDIT_FE_CREDIT_PATH = '/data/fpt/frt/fshop/dwh/fact_fe_credit_installment.parquet' # run_date
FRT_CREDIT_HOME_CREDIT_PATH = '/data/fpt/frt/fshop/dwh/fact_homecredit_installment.parquet' # run_date
FRT_CREDIT_MIRAE_ONL_PATH = '/data/fpt/frt/fshop/dwh/fact_mirae_credit_online_installment.parquet' # run_date
FRT_CREDIT_MIRAE_OFF_PATH = '/data/fpt/frt/fshop/dwh/fact_mirae_credit_offline_installment.parquet' # run_date
FRT_CREDIT_FSHOP_FORM_PATH = '/data/fpt/frt/fshop/dwh/fact_installment_registration_form.parquet' # run_date
FRT_CREDIT_FSHOP_CUSTOMER_PATH = '/data/fpt/frt/fshop/dwh/dim_fshop_installment_customer.parquet'# run_date
FRT_FSHOP_PATH = '/bigdata/fdp/frt/data/posdata/ict/pos_ocrd/' # month
FRT_LONGCHAU_PATH = '/bigdata/fdp/frt/data/posdata/pharmacy/posthuoc_ocrd/' # month
FSOFT_VIO_PATH = '/data/fpt/ftel/cads/dep_solution/sa/cdp/core_profile/get_active_status/fsoft_vio/dim_user.parquet' # p_date - 1(all)
FTEL_FPLAY_PATH = '/data/fpt/ftel/fplay/dwh/dim_user.parquet' # run_date
FTEL_INTERNET_DEMO_PATH = '/data/fpt/ftel/cads/dep_solution/sa/ftel/internet/data/rst_demographics.parquet' # run_date
FTEL_INTERNET_MULTI_PATH = '/data/fpt/ftel/cads/dep_solution/sa/ftel/internet/data/rst_multiphone.parquet' # run_date
SENDO_SENDO_PATH = '/data/fpt/sendo/dwh/stag_sale_order.parquet' # p_date

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

# ? PHONE DICT
RAW_PHONE_COLS = {
    'fo_vne': ['phone'],
    'frt_credit_fe_credit': ['customer_phone'],
    'frt_credit_home_credit': ['customer_phone', 'original_phone'],
    'frt_credit_mirae_onl': ['phone'],
    'frt_credit_mirae_off': [
        'customer_phone',
        'contact_person_phone',
        'contact_person_2_phone'
    ],
    'frt_credit_fshop_form': [
        'customer_phone',
        'contact_person_phone',
        'receiver_phone'
    ],
    'frt_credit_fshop_customer': [
        'phone',
        'contact_person_phone',
        'contact_person_2_phone',
        'contact_person_3_phone',
        'contact_person_4_phone',
        'valid_phone',
        'invalid_phone'
    ],
    'frt_fshop': [
        'PhoneDecrypt',
        'Phone'
    ],
    'frt_longchau': [
        'PhoneDecrypt',
        'Phone'
    ],
    'fsoft_vio': ['phone'],
    'ftel_fplay': ['phone'],
    'ftel_internet_demo': ['phone'],
    'ftel_internet_multi': ['phone'],
    'sendo_sendo': [
        'shipping_contact_phone',
        'buyer_phone',
        'store_phone'
    ]
}

# ? NAME DICT
RAW_NAME_COLS = {
    'fo_vne': ['name'],
    'frt_credit_fe_credit': ['customer_name', 'cc_name'],
    'frt_credit_fshop_customer': [
        'customer_name',
        'contact_person_name',
        'contact_person_2_name',
        'contact_person_3_name',
        'contact_person_4_name'
    ],
    'frt_credit_fshop_form': [
        'contact_person_name',
        'customer_name',
        'receiver_name'
    ],
    'frt_credit_home_credit': ['customer_name', 'pg_process_user_name'],
    'frt_credit_mirae_off': [
        'company_name',
        'contact_person_name',
        'contact_person_2_name',
        'last_name',
        'middle_name',
        'first_name'
    ],
    'frt_credit_mirae_onl': ['full_name'],
    'frt_fshop': ['CardName'],
    'frt_longchau': ['CardName'],
    'fsoft_vio': ['full_name', 'school_name'],
    'ftel_fplay': ['name'],
    'ftel_internet_demo': ['customer_name'],
    'ftel_internet_multi': ['contact_name', 'description'],
    'sendo_sendo': ['receiver_name', 'buyer_name']
}

VIETNAMESE_WORD_REGEX = "(?i)^[AĂÂÁẮẤÀẰẦẢẲẨÃẴẪẠẶẬĐEÊÉẾÈỀẺỂẼỄẸỆIÍÌỈĨỊOÔƠÓỐỚÒỒỜỎỔỞÕỖỠỌỘỢUƯÚỨÙỪỦỬŨỮỤỰYÝỲỶỸỴAĂÂÁẮẤÀẰẦẢẲẨÃẴẪẠẶẬĐEÊÉẾÈỀẺỂẼỄẸỆIÍÌỈĨỊOÔƠÓỐỚÒỒỜỎỔỞÕỖỠỌỘỢUƯÚỨÙỪỦỬŨỮỤỰYÝỲỶỸỴAĂÂÁẮẤÀẰẦẢẲẨÃẴẪẠẶẬĐEÊÉẾÈỀẺỂẼỄẸỆIÍÌỈĨỊOÔƠÓỐỚÒỒỜỎỔỞÕỖỠỌỘỢUƯÚỨÙỪỦỬŨỮỤỰYÝỲỶỸỴAĂÂÁẮẤÀẰẦẢẲẨÃẴẪẠẶẬĐÐEÊÉẾÈỀẺỂẼỄẸỆIÍÌỈĨỊOÔƠÓỐỚÒỒỜỎỔỞÕỖỠỌỘỢUƯÚỨÙỪỦỬŨỮỤỰYÝỲỶỸỴAĂÂÁẮẤÀẰẦẢẲẨÃẴẪẠẶẬĐEÊÉẾÈỀẺỂẼỄẸỆIÍÌỈĨỊOÔƠÓỐỚÒỒỜỎỔỞÕỖỠỌỘỢUƯÚỨÙỪỦỬŨỮỤỰYÝỲỶỸỴAĂÂÁẮẤÀẰẦẢẲẨÃẴẪẠẶẬĐEÊÉẾÈỀẺỂẼỄẸỆIÍÌỈĨỊOÔƠÓỐỚÒỒỜỎỔỞÕỖỠỌỘỢUƯÚỨÙỪỦỬŨỮỤỰYÝỲỶỸỴA-Z\s]+$"


# ? EMAIL DICT
RAW_EMAIL_COLS = {
    'fo_vne': ['email'],
    'frt_credit_fe_credit': [],
    'frt_credit_fshop_customer': ['email'],
    'frt_credit_fshop_form': [],
    'frt_credit_home_credit': [],
    'frt_credit_mirae_off': [],
    'frt_credit_mirae_onl': [],
    'frt_fshop': ['Email'],
    'frt_longchau': ['Email'],
    'fsoft_vio': ['email'],
    'ftel_fplay': ['email'],
    'ftel_internet_demo': ['email'],
    'ftel_internet_multi': [],
    'sendo_sendo': ['store_email', 'buyer_email']
}
DOMAIN_DICT_PATH = '/data/fpt/ftel/cads/dep_solution/user/quangvm9/helper/migrate/rst/email/domain_dict.parquet'
