
import datetime
import sys
sys.path.insert(1, '/bigdata/fdp/cdp/script/')

from config.cdp import *
from tqdm import tqdm
from glob import glob
from pyspark.sql.types import StructType, StructField, StringType, DateType, IntegerType, DecimalType, LongType
from pyspark.sql.window import Window as W

TODAY = str(sys.argv[1])[:10]
YESTERDAY = str(pd.to_datetime(TODAY) - pd.DateOffset(days=1))[:10]

COOKIE_PATH = config.get('fid', 'path', 'cookie')
IP_PATH = config.get('fid', 'path', 'ip')

ROOT_PATH = '/data/fpt/ftel/cads/dep_solution/sa/cdp/core'
PRE_PROFILE_PATH = ROOT_PATH + '/pre/{}.parquet/d={}'
PRE_NEW_PROFILE_PATH = ROOT_PATH + '/pre/{}_newcomer.parquet'

PROFILE_SCHEMA = StructType([
    StructField('fid', LongType(), nullable=False),
    StructField('key', StringType(), nullable=False),
    StructField('id', StringType(), nullable=False),
    StructField('phone', StringType(), nullable=True),
    StructField('email', StringType(), nullable=True),
    StructField('cdp_id', StringType(), nullable=True),
    StructField('ip', StringType(), nullable=True),
    StructField('fpartner', StringType(), nullable=False),
    StructField('version_date', DateType(), nullable=True)
])

FNAMES = ['fo', 'fplay', 'ftel', 'fshop', 'longchau', 'sendo']
FNAMES2KEYS = dict(zip(FNAMES, ['vne_id_fo', 'user_id_fplay', 'contract_ftel', 'cardcode_fshop', 'cardcode_longchau', 'id_sendo']))

# FID_PATH = ROOT_PATH + '/fid/test/'
# CHECKPOINT_PATH = os.path.join(FID_PATH, 'checkpoint')

FID_PATH = ROOT_PATH + '/fid/'
CHECKPOINT_PATH = ROOT_PATH + '/checkpoint'


##################################################################################################
##################################################################################################
def get_ip(fpartner, date):
    
    ftel_latest_day = sorted(os.listdir('/bigdata/fdp/user/mainhx/adhoc/ftel_fplay/ip_modem_ftel_by_day.parquet'))[-1]    
    ip_ftel = spark.read.parquet(
        f'file:///bigdata/fdp/user/mainhx/adhoc/ftel_fplay/ip_modem_ftel_by_day.parquet/{ftel_latest_day}'
    )
    mac_to_contract = spark.read.parquet(
        '/data/fpt/ftel/cads/dep_solution/user/namdp11/mapping_ip/data/ftel_mac_dict.parquet'
    )
    ip_ftel = ip_ftel.join(mac_to_contract, on='mac', how='inner')
    ip_contract = ip_ftel.groupby('ip').agg(F.countDistinct('contract').alias('ip_contract'))
    contract_ip = ip_ftel.groupby('contract').agg(F.countDistinct('ip').alias('contract_ip'))

    ip_ftel = (
        ip_ftel
        .join(ip_contract, on='ip')
        .join(contract_ip, on='contract')
        .filter((F.col('ip_contract') == 1) & (F.col('contract_ip') == 1))
        .selectExpr('ip', 'contract as id')
    )
    if fpartner == 'ftel':
        return ip_ftel
    
    elif fpartner == 'fplay':
        fplay_latest_file = sorted([f.path for f in hdfs.get_file_info(fs.FileSelector("/data/fpt/ftel/cads/dep_solution/user/namdp11/mapping_ip/data/daily/fplay/"))])[-1]
        ip = spark.read.parquet(
#             f'/data/fpt/ftel/cads/dep_solution/user/namdp11/mapping_ip/data/daily/fplay/fplay_ftel_{str(date)[:10]}.parquet',
            fplay_latest_file
        )
        ip_user_id = ip.groupby('ip').agg(F.countDistinct('user_id').alias('ip_user_id'))
        user_id_ip = ip.groupby('user_id').agg(F.countDistinct('ip').alias('user_id_ip'))
        ip = (
            ip
            .join(ip_user_id, on='ip')
            .join(user_id_ip, on='user_id')
            .join(ip_ftel.select('ip'), on='ip', how='inner')
            .filter((F.col('ip_user_id') == 1) & (F.col('user_id_ip') == 1) & F.col('is_box'))
            .selectExpr('ip', 'user_id as id')
        )
    elif fpartner == 'fo':
        fo_latest_file = sorted([f.path for f in hdfs.get_file_info(fs.FileSelector("/data/fpt/ftel/cads/dep_solution/user/namdp11/mapping_ip/data/daily/fo/"))])[-1]
        ip = spark.read.parquet(
#             f'/data/fpt/ftel/cads/dep_solution/user/namdp11/mapping_ip/data/daily/fo/fo_ftel_{str(date)[:10]}.parquet',
            fo_latest_file
        )
        ip_user_id = ip.groupby('ip').agg(F.countDistinct('user_id').alias('ip_user_id'))
        user_id_ip = ip.groupby('user_id').agg(F.countDistinct('ip').alias('user_id_ip'))
        ip = (
            ip
            .join(ip_user_id, on='ip')
            .join(user_id_ip, on='user_id')
            .join(ip_ftel.select('ip'), on='ip', how='inner')
            .filter((F.col('ip_user_id') == 1) & (F.col('user_id_ip') == 1))
            .selectExpr('ip', 'user_id as id')
        )
    else:
        raise f'{fpartner} does not have IP recorded.'
    ip = (
        ip
        .withColumn('fpartner', F.lit(fpartner))
        .withColumn('id', F.concat(F.upper('fpartner'), F.lit('-'), F.col('id')))
    )
    return ip
    
def get_cookie(date):
    #cid = spark.read.parquet(COOKIE_PATH.format(str(date)[:10]))
    
    #Bo comment dong tiep theo de doc file 27-07.
#     cid = spark.read.parquet(COOKIE_PATH.format('2022-07-27'))
    latest_cookie_path = sorted([f.path for f in hdfs.get_file_info(fs.FileSelector('/data/fpt/ftel/cads/dep_solution/user/namdp11/script_cdp/refactor/output')) if 'final_fid_login_cdp_' in f.path])[-1]
    cid = spark.read.parquet(latest_cookie_path)
    
    cid = (
        cid
        .withColumnRenamed('user_id', 'id')
        .withColumnRenamed('f_group', 'fpartner')
        .withColumn(
            'id',
            F.when(F.col('fpartner') == 'frt', F.regexp_replace('id', '-fshop', '')).otherwise(F.col('id'))
        )
        #.withColumn('root_id', F.col('id'))
        .withColumn('id', F.concat(F.upper('fpartner'), F.lit('-'), F.col('id')))
    )
    return cid

def save(df, path):
    (
        df
        .repartition(1)
        .write.partitionBy('d')
        .mode('overwrite')
        .option('partitionOverwriteMode', 'dynamic')
        .parquet(path)
    )

def preprocess(profile, fpartner, date): 
    cid = get_cookie(date)
    id_col = FNAMES2KEYS[fpartner]

    profile = (
        profile
        .withColumn('fpartner', F.lit(fpartner))
        .withColumn('root_id', F.col(id_col))
        .withColumn(id_col, F.concat(F.upper('fpartner'), F.lit('-'), F.col(id_col)))
        .withColumn('d', F.lit(str(date)[:10]))
        .join(cid.selectExpr(f'id as {id_col}', 'cdp_id'), on=id_col, how='left')
    )
        
    if fpartner in ['ftel', 'fplay', 'fo']:
        ip = get_ip(fpartner, date)
        profile = (
            profile
            .join(ip.selectExpr(f'id as {id_col}', 'ip'), on=id_col, how='left'))
    else:
        profile = (
            profile
            .withColumn('ip', F.lit(None))
            .withColumn('ip', F.col('ip').cast(StringType()))
        )
    profile = profile.withColumn(
        'key', 
        F.concat_ws('+', F.col(id_col), F.col('phone'), F.col('email'), F.col('cdp_id'), F.col('ip'))
    )
    
    return profile.select('key', id_col, 'phone', 'email', 'cdp_id', 'ip', 'fpartner', 'is_email_valid', 'is_phone_valid')


if __name__ == '__main__':
    last_profiles = spark.read.parquet(
        os.path.join(FID_PATH, 'fid_core_profile.parquet', f'd={YESTERDAY}')
    )

    last_profiles = (
        last_profiles
        .drop('fid_prev')
        .withColumn('ip', F.lit(None))
        .withColumn('ip', F.col('ip').cast(StringType()))
        .withColumn(
            'key', 
            F.concat_ws('+', F.col('id'), F.col('phone'), F.col('email'), F.col('cdp_id'))
        )
    )
    
    pbar = tqdm(FNAMES)
    for fpartner in pbar:
        pbar.set_description(f'fpartner={fpartner}')
        f_profile = spark.read.parquet(PRE_PROFILE_PATH.format(fpartner, TODAY))
        f_profile = (
            preprocess(
                (
                    f_profile
                    .filter((F.col('is_email_valid') == True) | (F.col('is_phone_valid') == True))
                    .withColumn('email', F.when(F.col('is_email_valid') == False, F.lit(None)).otherwise(F.col('email')))
                    .withColumn('phone', F.when(F.col('is_phone_valid') == False, F.lit(None)).otherwise(F.col('phone')))
                ),
                fpartner,
                TODAY
            )
            # Left-anti on key.
            .join(last_profiles, on='key', how='leftanti')
            .withColumn('d', F.lit(TODAY))
            .dropDuplicates(subset=['key'])
        )

        #save(
        #    f_profile, 
        #    PRE_NEW_PROFILE_PATH.format(fpartner),
        #      partitions=1
        #)
        
        (f_profile
        .repartition(1)
        .write.partitionBy('d')
        .mode('overwrite')
        .option('partitionOverwriteMode', 'dynamic')
        .parquet(PRE_NEW_PROFILE_PATH.format(fpartner)))
    
    for fpartner in FNAMES:
        try:
            temp = spark.read.parquet(
                os.path.join(PRE_NEW_PROFILE_PATH.format(fpartner), f'd={TODAY}')
            )
            print(fpartner, temp.count())
        except:
            print(f'[WARNING] No new profile found for {fpartner}')
            continue
