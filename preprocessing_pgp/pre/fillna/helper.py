
from utils import *
import pickle
import string

ROOT_PATH = '/data/fpt/ftel/cads/dep_solution/sa/cdp/core'

FTEL_SA_PATH = '/data/fpt/ftel/cads/dep_solution/sa/ftel'
PHONE_CAMPAIGN = f'{FTEL_SA_PATH}/internet/data/phone_active.parquet'

ACCENT_MODEL = ROOT_PATH + '/utils/name2gender/accented/logistic_pipeline.pkl'
NO_ACCENT_MODEL = ROOT_PATH + '/utils/name2gender/not_accented/logistic_pipeline.pkl'

FO_LOCATION = ROOT_PATH + '/utils/fo_location_most.parquet'
FPLAY_LOCATION = ROOT_PATH + '/utils/fplay_location_most.parquet'

def rename_columns(df, columns):
    if isinstance(columns, dict):
        return df.select(*[F.col(col_name).alias(columns.get(col_name, col_name)) for col_name in df.columns])
    else:
        raise ValueError("'columns' should be a dict")

def save_data(df, date, save_path=''):
    (df.withColumn('d', F.lit(date.strftime('%F')))
     .repartition(1).write.partitionBy('d')
     .mode("overwrite").option("partitionOverwriteMode", "dynamic")
     .parquet(save_path))
        
def join_multifield(df1, df2, keys=list(), how='inner'):
    result = (
        df1.fillna('no_infor', subset=keys)
        .join(df2.fillna('no_infor', subset=keys), on=keys, how=how)
    )
    return result.select(*[F.when(F.col(c) == 'no_infor', F.lit(None)).otherwise(F.col(c)).alias(c) for c in result.columns])

def remove_accents(df, colname):
    return (
        df.withColumn(colname, F.regexp_replace(F.col(colname), '[àáạảãâầấậẩẫăằắặẳẵ]', 'a'))
        .withColumn(colname, F.regexp_replace(F.col(colname), '[ÀÁẠẢÃĂẰẮẶẲẴÂẦẤẬẨẪ]', 'A'))
        .withColumn(colname, F.regexp_replace(F.col(colname), '[èéẹẻẽêềếệểễ]', 'e'))
        .withColumn(colname, F.regexp_replace(F.col(colname), '[ÈÉẸẺẼÊỀẾỆỂỄ]', 'E'))
        .withColumn(colname, F.regexp_replace(F.col(colname), '[òóọỏõôồốộổỗơờớợởỡ]', 'o'))
        .withColumn(colname, F.regexp_replace(F.col(colname), '[ÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠ]', 'O'))
        .withColumn(colname, F.regexp_replace(F.col(colname), '[ìíịỉĩ]', 'i'))
        .withColumn(colname, F.regexp_replace(F.col(colname), '[ÌÍỊỈĨ]', 'I'))
        .withColumn(colname, F.regexp_replace(F.col(colname), '[ùúụủũưừứựửữ]', 'u'))
        .withColumn(colname, F.regexp_replace(F.col(colname), '[ƯỪỨỰỬỮÙÚỤỦŨ]', 'U'))
        .withColumn(colname, F.regexp_replace(F.col(colname), '[ỳýỵỷỹ]', 'y'))
        .withColumn(colname, F.regexp_replace(F.col(colname), '[ỲÝỴỶỸ]', 'Y'))
        .withColumn(colname, F.regexp_replace(F.col(colname), '[Đ]', 'D'))
        .withColumn(colname, F.regexp_replace(F.col(colname), '[đ]', 'd'))
    )

# Model Gender

def load_model(file_path, module='pickle'):
    
    if module == 'pickle':
        model = pickle.loads(hdfs.open_input_file(file_path).read())
        
    return model

def preprocessing_name(df_profile, name_col='name', max_length=None):
    
    df_profile['clean_name'] = df_profile[name_col].str.lower().str.replace(
        r'\d', ' ').str.replace(
        rf'[{string.punctuation}]', ' ').str.replace(
        r'\s+', ' ').str.strip()
    
    return df_profile

def get_gender_from_name(df_profile, type_name='accent', name_col='name'):
    
    df_prep = preprocessing_name(df_profile, name_col)
    
    if type_name=='accent':
        path = ACCENT_MODEL
    elif type_name=='no_accent':
        path = NO_ACCENT_MODEL
    
    pipeline = load_model(path)
    
    predictions = pipeline.predict(df_prep['clean_name'].values)
    df_prep['gender_prediction'] = list(map(lambda x: 'M' if x == 1 else 'F', predictions))
    
    return df_prep

def get_address(spark, id_cttv='vne_id_fo', path=''):
    
    df_address = (
        spark.read.parquet(path)
        .drop('num_date')
        .withColumn('address_ip', F.concat_ws(', ', 'name_district', 'name_province'))
        .withColumnRenamed('user_id', id_cttv)
    )
    
    return df_address
    
def fill_location(spark, date, input_path='', save_path='', id_cttv='vne_id_fo'):
    
    start = time()
    
    if id_cttv == 'vne_id_fo':
        # Load IP Location
        df_address = get_address(spark, id_cttv=id_cttv, path=FO_LOCATION)
        
    if id_cttv == 'user_id_fplay':
        # Load IP Location
        df_address = get_address(spark, id_cttv=id_cttv, path=FPLAY_LOCATION)
        
    df_profile = spark.read.parquet(input_path + '/d={}'.format(date.strftime('%F')))
    print('Shape before: {}'.format(df_profile.count()))
        
    df_profile = (
        df_profile
        .join(df_address, on=id_cttv, how='left')
        # fillna address
        .withColumn('address', F.coalesce('address', 'address_ip'))
        .withColumn('source_address', F.when(F.col('address').isNotNull() & F.col('source_address').isNull(), F.lit('From IP')).otherwise(F.col('source_address')))
        # fillna city
        .withColumn('city', F.coalesce('city', 'name_province'))
        .withColumn('source_city', F.when(F.col('address').isNotNull() & F.col('source_city').isNull(), F.lit('From IP')).otherwise(F.col('source_city')))
        # fillna district
        .withColumn('district', F.coalesce('district', 'name_district'))
        .withColumn('source_district', F.when(F.col('address').isNotNull() & F.col('source_district').isNull(), F.lit('From IP')).otherwise(F.col('source_district')))
        .drop('address_ip', 'name_province', 'name_district')
    )
    
    print('Shape after: {}'.format(df_profile.count()))
    
    save_data(df_profile, date, save_path=save_path)
    
    stop = time()

    print(f'Time for fillna location: {int(stop - start)}s')


def fill_email(spark, date, input_path='', save_path=''):
    
    start = time()
    # Dict Email
    df_email2profile = (
        spark.read.parquet(ROOT_PATH + '/utils/valid_email_latest.parquet')
        .filter(F.col('username_iscertain'))
        .filter(F.col('email').isNotNull())
        .withColumnRenamed('address', 'city')
        .select('email', 'username', 'year_of_birth', 'city')
        .dropDuplicates(subset=['email'])
        .withColumn('year_of_birth', F.col('year_of_birth').cast('int'))
        .withColumn(
            'birthday', 
            F.when(F.col('year_of_birth').isNotNull(), F.concat(F.lit('0/0/'), F.col('year_of_birth').cast('string')))
        )
        .drop('year_of_birth')
        .cache()
    )
    print(df_email2profile.count())

    df_email2profile = rename_columns(df_email2profile, {
        'username': 'username_email',
        'birthday': 'birthday_email',
        'city': 'city_email',
    })
    
    df_profile = spark.read.parquet(input_path + '/d={}'.format(date.strftime('%F')))
    print('Shape before: {}'.format(df_profile.count()))

    df_profile = (
        df_profile
        .join(df_email2profile, on='email', how='left')
        # fillna name
        .withColumn('name', F.coalesce('name', 'username_email'))
        .withColumn('source_name', F.when(F.col('name').isNotNull() & F.col('source_name').isNull(), F.lit('From Email')).otherwise(F.col('source_name')))
        # fillna birthday
        .withColumn('birthday', F.coalesce('birthday', 'birthday_email'))
        .withColumn('source_birthday', F.when(F.col('birthday').isNotNull() & F.col('source_birthday').isNull(), F.lit('From Email')).otherwise(F.col('source_birthday')))
        # fillna city
        .withColumn('city', F.coalesce('city', 'city_email'))
        .withColumn('source_city', F.when(F.col('city').isNotNull() & F.col('source_city').isNull(), F.lit('From Email')).otherwise(F.col('source_city')))
        .drop('username_email', 'city_email', 'birthday_email')
    )
    
    print('Shape after: {}'.format(df_profile.count()))

    save_data(df_profile, date, save_path=save_path)
    
    stop = time()

    print(f'Time for fillna email: {int(stop - start)}s')

def fill_gender(spark, date, input_path='', save_path=''):
    
    start = time()
    
    # pronoun-gender dict
    pronoun_gender_dict = spark.read.parquet(ROOT_PATH + '/fillna/pronoun_gender_dict.parquet')

    # add column idx
    df_profile = spark.read.parquet(input_path + '/d={}'.format(date.strftime('%F'))).withColumn("idx", F.monotonically_increasing_id())
    print('Shape before: {}'.format(df_profile.count()))
    
    df_profile_with_pronoun = df_profile.filter(F.col('pronoun').isNotNull())
    df_profile_with_non_pronoun = df_profile.filter(F.col('pronoun').isNull())
    
    # profile with pronoun
    df_profile_with_pronoun = (
        df_profile_with_pronoun
        .join(pronoun_gender_dict.withColumnRenamed('gender', 'gender_pronoun'), on='pronoun', how='left')
        .withColumn('gender', F.coalesce('gender_pronoun', 'gender'))
        .drop('idx', 'gender_pronoun')
    )
    
    # profile without pronoun
    # get all name with no gender
    df_all_name = (
        df_profile_with_non_pronoun
        .filter(F.col('gender').isNull())
        .filter(F.col('name').isNotNull())
        .filter((F.col('customer_type') == 'Ca nhan')|(F.col('customer_type').isNull()))
        .select('idx', 'name')
        .toPandas()
    )
    
    # predict gender by accent name
    condition_accent = df_all_name.name.notna() & (df_all_name.name.str.encode('ascii', errors='ignore') != df_all_name.name.str.encode('ascii', errors='replace'))
    df_all_name_accent = df_all_name[condition_accent]

    gender_predict_accent = get_gender_from_name(df_all_name_accent, type_name='accent', name_col='name')

    gender_predict_accent = spark.createDataFrame(gender_predict_accent[['idx', 'gender_prediction']])
    df_profile_with_non_pronoun = (
        df_profile_with_non_pronoun
        .join(gender_predict_accent, on='idx', how='left')
        .withColumn('gender', F.coalesce('gender', 'gender_prediction'))
        .withColumn(
            'source_gender', 
            F.when(F.col('gender').isNotNull() & F.col('source_gender').isNull(), 'Name2Gender')
            .otherwise(F.col('source_gender'))
        )
        .drop('gender_prediction')
    )

    # predict gender by non-accent name
    condition_non_accent = df_all_name.name.notna() & (df_all_name.name.str.encode('ascii', errors='ignore') == df_all_name.name.str.encode('ascii', errors='replace'))
    df_all_name_non_accent = df_all_name[condition_non_accent]

    gender_predict_non_accent = get_gender_from_name(df_all_name_non_accent, type_name='no_accent', name_col='name')
    gender_predict_non_accent = spark.createDataFrame(gender_predict_non_accent[['idx', 'gender_prediction']])

    df_profile_with_non_pronoun = (
        df_profile_with_non_pronoun
        .join(gender_predict_non_accent, on='idx', how='left')
        .withColumn('gender', F.coalesce('gender', 'gender_prediction'))
        .withColumn(
            'source_gender', 
            F.when(F.col('gender').isNotNull() & F.col('source_gender').isNull(), 'Name2Gender')
            .otherwise(F.col('source_gender'))
        )
        .drop('idx', 'gender_prediction')
    )

    df_profile = df_profile_with_pronoun.unionByName(df_profile_with_non_pronoun).cache()

    print('Shape after: {}'.format(df_profile.count()))
    
    # Save data
    save_data(df_profile, date, save_path=save_path)
    
    stop = time()

    print(f'Time for fillna gender: {int(stop - start)}s')
    
def fillna_pipeline(spark, date, input_path='', save_path='', id_cttv='vne_id_fo'):
    
    # fill ip location
    if id_cttv == 'user_id_fplay' or id_cttv == 'vne_id_fo':
        fill_location(spark, date, input_path=input_path, save_path=save_path, id_cttv=id_cttv)
        fill_email(spark, date, input_path=save_path, save_path=save_path)
        fill_gender(spark, date, input_path=save_path, save_path=save_path)
    else:
        fill_email(spark, date, input_path=input_path, save_path=save_path)
        fill_gender(spark, date, input_path=save_path, save_path=save_path)
    
def post_processing(spark, date, path='', cttv='vne_id_fo'):
     
    start = time()

    df_profile = spark.read.parquet(path + '/d={}'.format(date.strftime('%F')))

    # Fillna customer_type
    df_profile = (
        df_profile
        .withColumn(
            'customer_type', 
            F.when(F.col('name').isNotNull() & F.col('customer_type').isNull(), F.lit('Ca nhan'))
            .otherwise(F.col('customer_type'))
        )
        .withColumn(
            'source_customer_type', 
            F.when(F.col('customer_type').isNotNull() & F.col('source_customer_type').isNull(), F.lit(cttv))
            .otherwise(F.col('source_customer_type'))
        )
        .withColumn('gender', F.when(F.col('customer_type') != 'Ca nhan', lit(None)).otherwise(F.col('gender')))
        .withColumn('source_gender', F.when(F.col('customer_type') != 'Ca nhan', lit(None)).otherwise(F.col('source_gender')))
    )
    
    # source Phone(only for FTEL)
    if cttv == 'FTEL':
        latest_file = sorted([f.path for f in hdfs.get_file_info(fs.FileSelector(ROOT_PATH + '/pre/ftel.parquet'))])[-1]
        ftel_source = spark.read.parquet(latest_file).withColumn('contract_phone_ftel', F.concat(F.col('contract_ftel'), lit('-'), F.col('phone'))).select('contract_phone_ftel', 'source')

        df_profile = (
            df_profile
            .join(
                ftel_source
                .filter(F.col('source') == 'demo')
                .select('contract_phone_ftel')
                .withColumn('is_demo', lit(True)), 
                on='contract_phone_ftel', 
                how='left')
            .withColumn(
                'source_phone', 
                F.when((F.col('source_phone') == 'FTEL') & (F.col('is_demo') == True), F.lit('FTEL from demo'))
                .when((F.col('source_phone') == 'FTEL') & (F.col('is_demo').isNull()), F.lit('FTEL from multi'))
                .otherwise(F.col('source_phone'))
            )
        )
    # source Phone(only for SENDO)    
    if cttv == 'SENDO':
        latest_file = sorted([f.path for f in hdfs.get_file_info(fs.FileSelector(ROOT_PATH + '/pre/sendo.parquet'))])[-1]
        sendo_source = spark.read.parquet(latest_file).withColumn('id_phone_sendo', F.concat(F.col('id_sendo'), lit('-'), F.col('phone'))).select('id_phone_sendo', 'source')
        
        df_profile = (
            df_profile
            .join(
                sendo_source
                .filter(F.col('source') == 'buyer')
                .select('id_phone_sendo')
                .withColumn('is_buyer', F.lit(True)), 
                on='id_phone_sendo', 
                how='left')
            .withColumn(
                'source_phone', 
                F.when((F.col('source_phone') == 'SENDO') & (F.col('is_buyer') == True), F.lit('SENDO from buyer'))
                .when((F.col('source_phone') == 'SENDO') & (F.col('is_buyer').isNull()), F.lit('SENDO from receiver'))
                .otherwise(F.col('source_phone'))
            )
        )

    # Fix address
    FILLNA_COLS = ['unit_address', 'ward', 'district', 'city']
    DICT_TRASH = {'': None, 'Nan': None, 'nan': None, 
              'None': None, 'none': None, 'Null': None, 'null': None, "''": None}


    df_profile = (
        df_profile
        .withColumn(
            'address', 
            F.concat_ws(', ', F.col('unit_address'), F.col('ward'), F.col('district'), F.col('city'))
        )
        .withColumn('address', F.regexp_replace(F.col('address'), '(?<![a-zA-Z0-9]),', ''))
        .withColumn('address', F.regexp_replace(F.col('address'), '-(?![a-zA-Z0-9])', ''))
        .withColumn('address', F.trim(F.col('address')))
    )

    df_profile = df_profile.replace(DICT_TRASH, subset=['address'])           
    df_profile = (
        df_profile
        .withColumn(
            'source_address', 
            F.when(F.col('address').isNotNull() & F.col('source_address').isNull(), F.col('source_city'))
            .when(F.col('address').isNull(), lit(None))
            .otherwise(F.col('source_address')))
    )

    # Fillna address if not have all 'ward', 'district', 'city'
    df_profile = (
        df_profile
        .withColumn(
            'num_filled_address', 
            reduce(add, [F.when(F.col(c).isNotNull(), F.lit(1)).otherwise(F.lit(0)) for c in ['ward', 'district', 'city']]))
        .withColumn('address', F.when(F.col('num_filled_address') != 3, F.lit(None)).otherwise(F.col('address')))
        .withColumn('source_address', F.when(F.col('num_filled_address') != 3, F.lit(None)).otherwise(F.col('source_address')))
        .drop('num_filled_address')
        .dropDuplicates()
    )
    # Saving data
    save_data(df_profile, date, save_path=path)

    stop = time()

    print(f'Time for postprocessing data: {int(stop - start)}')
    
def generate_total_id(spark, date):
    
    start = time()
    
    fids = (
        spark.read.parquet(ROOT_PATH + '/fid/fid_core_score.parquet/d={}'.format(date.strftime("%F")))
        .filter(F.col('is_abnormal') == False)
        .select('fid', 'fid_score')
        .dropDuplicates())
    
    fid_fo = spark.read.parquet(ROOT_PATH + '/fid/fid_fo.parquet/d={}'.format(date.strftime("%F")))
    fid_fo = fid_fo.join(fids.select('fid'), on='fid', how='inner').select('fid', 'vne_id_fo').dropDuplicates()
    
    fid_fplay = spark.read.parquet(ROOT_PATH + '/fid/fid_fplay.parquet/d={}'.format(date.strftime("%F")))
    fid_fplay = fid_fplay.join(fids.select('fid'), on='fid', how='inner').select('fid', 'user_id_fplay').dropDuplicates()
    
    fid_fshop = (
        spark.read.parquet(ROOT_PATH + '/fid/fid_fshop.parquet/d={}'.format(date.strftime("%F")))
        .join(fids.select('fid'), on='fid', how='inner')
        .withColumn('cardcode_fshop', F.col('cardcode_fshop').cast('string'))
        .filter(~F.col('cardcode_fshop').rlike('-'))
        .select('fid', 'cardcode_fshop')
        .dropDuplicates()
    )
    
    fid_longchau = (
        spark.read.parquet(ROOT_PATH + '/fid/fid_longchau.parquet/d={}'.format(date.strftime("%F")))
        .join(fids.select('fid'), on='fid', how='inner')
        .withColumn('cardcode_longchau', F.col('cardcode_longchau').cast('string'))
        .filter(~F.col('cardcode_longchau').rlike('-'))
        .select('fid', 'cardcode_longchau')
        .dropDuplicates()
    )
    
    fid_sendo = spark.read.parquet(ROOT_PATH + '/fid/fid_sendo.parquet/d={}'.format(date.strftime("%F")))
    fid_sendo = fid_sendo.join(fids.select('fid'), on='fid', how='inner').select('fid', 'id_sendo', 'phone')
    fid_sendo = (
        fid_sendo
        .withColumn(
            'id_phone_sendo', 
            F.when((F.col('id_sendo').isNotNull()) & (F.col('phone').isNotNull()), F.concat_ws('-', F.col('id_sendo'), F.col('phone')))
            .when((F.col('id_sendo').isNotNull()) & (F.col('phone').isNull()), F.col('id_sendo'))
        )
        .select('fid', 'id_phone_sendo')
        .dropDuplicates())
    
    fid_ftel = spark.read.parquet(ROOT_PATH + '/fid/fid_ftel.parquet/d={}'.format(date.strftime("%F")))
    fid_ftel = fid_ftel.join(fids.select('fid'), on='fid', how='inner').select('fid', 'contract_ftel', 'phone')
    fid_ftel = (
        fid_ftel
        .withColumn(
            'contract_phone_ftel', 
            F.when((F.col('contract_ftel').isNotNull()) & (F.col('phone').isNotNull()), F.concat_ws('-', F.col('contract_ftel'), F.col('phone')))
            .when((F.col('contract_ftel').isNotNull()) & (F.col('phone').isNull()), F.col('contract_ftel'))
        )
        .select('fid', 'contract_phone_ftel')
        .dropDuplicates())
    
    # add 2 more credit and fsoft
    fid_credit = spark.read.parquet(ROOT_PATH + '/fid/fid_credit.parquet/d={}'.format(date.strftime("%F")))
    fid_credit = (
        fid_credit
        .join(fids.select('fid'), on='fid', how='inner')
        .select('fid', 'cardcode_fshop')
        .withColumn('cardcode_fshop', F.col('cardcode_fshop').cast('string'))
        .dropDuplicates())
    
    fid_fsoft = spark.read.parquet(ROOT_PATH + '/fid/fid_fsoft.parquet/d={}'.format(date.strftime("%F")))
    fid_fsoft = fid_fsoft.join(fids.select('fid'), on='fid', how='inner').select('fid', 'fsoft_id').dropDuplicates()
    
    mapping_fids = (
        fids
        .join(fid_fo, on='fid', how='left')
        .join(fid_fplay, on='fid', how='left')
        .join(fid_fshop.unionByName(fid_credit), on='fid', how='left')
        .join(fid_longchau, on='fid', how='left')
        .join(fid_sendo, on='fid', how='left')
        .join(fid_ftel, on='fid', how='left')
        # .join(fid_credit, on='fid', how='left')
        .join(fid_fsoft, on='fid', how='left')
    )
    
    (mapping_fids
    .withColumn('d', F.lit(date.strftime('%F')))
    .repartition(1).write.partitionBy("d")
    .mode("overwrite").option("partitionOverwriteMode", "dynamic")
    .parquet(ROOT_PATH + '/utils/mapping_fids.parquet'))
    
    stop = time()
    
    print(f'Time for mapping_fids: {int(stop - start)}s')
    
def generate_profile_with_fid(spark, date, f_group, key_map):
    
    mapping_fids = spark.read.parquet(ROOT_PATH + '/utils/mapping_fids.parquet/d={}'.format(date.strftime('%F')))
    profile_non_fid = spark.read.parquet(f'{ROOT_PATH}/pre/{f_group}.parquet/d={date.strftime("%F")}')
    if 'email_raw' not in profile_non_fid.columns:
        profile_non_fid = (
            profile_non_fid
            .withColumn('email_raw', F.lit(None))
            .withColumn('email', F.lit(None))
            .withColumn('is_email_valid', F.lit(False))
        )
    if 'phone_raw' not in profile_non_fid.columns:
        profile_non_fid = (
            profile_non_fid
            .withColumn('phone_raw', F.lit(None))
            .withColumn('phone', F.lit(None))
            .withColumn('is_phone_valid', F.lit(False))
        )
    profile_non_fid = (
        profile_non_fid
        .withColumn('phone', F.when(F.col('is_phone_valid') != True, lit(None)).otherwise(F.col('phone')))
        .withColumn('email', F.when(F.col('is_email_valid') != True, lit(None)).otherwise(F.col('email'))))
    
    if f_group == 'fshop': # key_map: cardcode_fshop
        mapping_fids = mapping_fids.filter(F.col(key_map).isNotNull()).filter(~F.col(key_map).rlike('credit_')).dropDuplicates()
    elif f_group == 'credit': # key_map: cardcode_fshop
        mapping_fids = mapping_fids.filter(F.col(key_map).isNotNull()).filter(F.col(key_map).rlike('credit_')).dropDuplicates()
    else:
        mapping_fids = mapping_fids.filter(F.col(key_map).isNotNull()).dropDuplicates()
    profile = (
        profile_non_fid
        .drop('phone_raw', 'is_phone_valid', 'email_raw', 'is_email_valid') #, 'is_full_name', 'customer_type_detail')
        .filter(F.col(key_map).isNotNull())
        .dropDuplicates()
        .join(mapping_fids, on=key_map, how='right'))
    
    return profile
