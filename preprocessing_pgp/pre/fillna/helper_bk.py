
from utils import *
import pickle
import string

RAW_PROFILE_PATH = '/data/fpt/ftel/cads/dep_solution/sa/dev/raw/'

FID_PATH = '/data/fpt/ftel/cads/dep_solution/sa/dev/fid'
PREPROCESS_PATH = '/data/fpt/ftel/cads/dep_solution/sa/dev/utils'

UNIFY_PATH = '/data/fpt/ftel/cads/dep_solution/sa/dev/pre'
CROSSFILL_PATH = '/data/fpt/ftel/cads/dep_solution/sa/dev/pos/crossfill'
FILLNA_PATH = '/data/fpt/ftel/cads/dep_solution/sa/dev/pos/fillna'

PHONE_CAMPAIGN_PATH = '/data/fpt/ftel/cads/dep_solution/sa/ftel/internet/data/phone_active.parquet'
UNIQUE_PATH = '/data/fpt/ftel/cads/dep_solution/sa/dev/unique'

ACCENT_MODEL_PATH = '/data/fpt/ftel/cads/dep_solution/user/namdp11/name2gender/accented/logistic_pipeline.pkl'
NO_ACCENT_MODEL_PATH = '/data/fpt/ftel/cads/dep_solution/user/namdp11/name2gender/not_accented/logistic_pipeline.pkl'

FO_LOCATION_PATH = '/data/fpt/ftel/cads/dep_solution/user/namdp11/scross_fill/runner/refactor/material/fo_location_most.parquet'
FPLAY_LOCATION_PATH = '/data/fpt/ftel/cads/dep_solution/user/namdp11/scross_fill/runner/refactor/material/fplay_location_most.parquet'


def rename_columns(df, columns):
    if isinstance(columns, dict):
        return df.select(*[F.col(col_name).alias(columns.get(col_name, col_name)) for col_name in df.columns])
    else:
        raise ValueError("'columns' should be a dict")

# Model Gender


def load_model(file_path, module='pickle'):

    if module == 'pickle':
        model = pickle.loads(hdfs.open_input_file(file_path).read())

    return model


def preprocessing_name(df, name_col='name', max_length=None):

    df = df.copy()

    df['clean_name'] = df[name_col].str.lower().str.replace(
        r'\d', ' ').str.replace(
        rf'[{string.punctuation}]', ' ').str.replace(
        r'\s+', ' ').str.strip()

    return df


def get_gender_from_name(df, type_name='accent', name_col='name'):

    df_prep = preprocessing_name(df.copy(), name_col)

    if type_name == 'accent':
        path = ACCENT_MODEL_PATH
    elif type_name == 'no_accent':
        path = NO_ACCENT_MODEL_PATH

    pipeline = load_model(path)

    predictions = pipeline.predict(df_prep['clean_name'].values)
    predictions = list(map(lambda x: 'M' if x == 1 else 'F', predictions))

    return predictions


def fill_location(df, id_cttv):

    start = time()

    if id_cttv == 'vne_id_fo':
        # Load IP Location
        df_address = pd.read_parquet(FO_LOCATION_PATH, filesystem=hdfs)
        df_address = df_address.drop(columns=['num_date'])

        df_address.loc[df_address.name_province.notna(), 'address_ip'] = df_address.loc[df_address.name_province.notna(
        ), 'name_district'] + ', ' + df_address.loc[df_address.name_province.notna(), 'name_province']
        df_address = df_address.rename(columns={'user_id': 'vne_id_fo'})

        # Fillna location
        df_profile = df.copy()

        df_profile1 = df_profile[df_profile.address.isna()]
        df_profile2 = df_profile[df_profile.address.notna()]

        df_profile1 = df_profile1.merge(df_address, how='left', on='vne_id_fo')

    if id_cttv == 'user_id_fplay':
        # Load IP Location
        df_address = pd.read_parquet(FPLAY_LOCATION_PATH, filesystem=hdfs)
        df_address = df_address.drop(columns=['num_date'])

        df_address.loc[df_address.name_province.notna(), 'address_ip'] = df_address.loc[df_address.name_province.notna(
        ), 'name_district'] + ', ' + df_address.loc[df_address.name_province.notna(), 'name_province']
        df_address = df_address.rename(columns={'user_id': 'user_id_fplay'})

        # Fillna location
        df_profile = df.copy()

        df_profile1 = df_profile[df_profile.address.isna()]
        df_profile2 = df_profile[df_profile.address.notna()]

        df_profile1 = df_profile1.merge(
            df_address, how='left', on='user_id_fplay')

    df_profile1['address'] = df_profile1['address_ip']
    df_profile1['city'] = df_profile1['name_province']
    df_profile1['district'] = df_profile1['name_district']
    df_profile1.loc[df_profile1.address.notna(), 'source_address'] = 'From IP'
    df_profile1.loc[df_profile1.address.notna(), 'source_city'] = 'From IP'
    df_profile1.loc[df_profile1.address.notna(), 'source_district'] = 'From IP'

    df_profile1 = df_profile1.drop(
        columns=['address_ip', 'name_province', 'name_district'])
    df_profile = df_profile1.append(df_profile2)
    df_profile = df_profile.drop_duplicates().reset_index(drop=True)

    stop = time()

    print(f'Time for fillna location by ip: {int(stop - start)}s')

    return df_profile


def fill_email(df):

    start = time()

    # Dict Email
    df_email2profile = pd.read_parquet(
        '/data/fpt/ftel/cads/dep_solution/sa/cdp/data/email_dict_latest.parquet', filesystem=hdfs)
    df_email2profile.loc[df_email2profile.address.notna(
    ), 'city'] = df_email2profile.loc[df_email2profile.address.notna(), 'address']
    df_email2profile = df_email2profile[[
        'email', 'phone', 'username', 'year_of_birth', 'address', 'city']]
    df_email2profile.columns = ['email', 'phone_email', 'username_email',
                                'year_of_birth_email', 'address_email', 'city_email']
    df_email2profile.loc[df_email2profile['year_of_birth_email'].notna(), 'birthday_email'] = '0/0/' + \
        df_email2profile['year_of_birth_email'].astype(
            str).str.replace('.0', '', regex=False)
    df_email2profile = df_email2profile.drop_duplicates(
        subset=['email'], keep='last')

    # Fillna email
    df_profile = df.copy()
    df_profile = df_profile.merge(df_email2profile, how='left', on='email')

#     df_profile.loc[df_profile.phone.isna(), 'phone'] = df_profile['phone_email']
#     df_profile.loc[df_profile.phone.notna() & df_profile.source_phone.isna(), 'source_phone'] = 'From Email'

    df_profile.loc[df_profile.name.isna(
    ), 'name'] = df_profile['username_email']
    df_profile.loc[df_profile.name.notna(
    ) & df_profile.source_name.isna(), 'source_name'] = 'From Email'

    df_profile.loc[df_profile.birthday.isna(
    ), 'birthday'] = df_profile['birthday_email']
    df_profile.loc[df_profile.birthday.notna(
    ) & df_profile.source_birthday.isna(), 'source_birthday'] = 'From Email'

    df_profile.loc[df_profile.address.isna(
    ), 'address'] = df_profile['address_email']
    df_profile.loc[df_profile.address.notna(
    ) & df_profile.source_address.isna(), 'source_address'] = 'From Email'

    df_profile.loc[df_profile.city.isna(), 'city'] = df_profile['city_email']
    df_profile.loc[df_profile.city.notna(
    ) & df_profile.source_city.isna(), 'source_city'] = 'From Email'

    df_profile = df_profile.drop(columns=['phone_email', 'username_email',
                                 'year_of_birth_email', 'address_email', 'city_email', 'birthday_email'])

    stop = time()

    print(f'Time for fillna email: {int(stop - start)}s')

    return df_profile


def fill_gender(df, id_cttv, tmp_path):

    start = time()

    # Chaos gender
    df_profile = df.copy()
    stats_gender_chaos = df_profile.groupby(
        by=[id_cttv])['gender'].agg(num_gender='nunique').reset_index()
    stats_gender_chaos = stats_gender_chaos[stats_gender_chaos.num_gender == 2]

    gender_chaos = df_profile[df_profile[id_cttv].isin(
        stats_gender_chaos[id_cttv])]
    gender_chaos = gender_chaos[gender_chaos.gender.notna()]

    if gender_chaos.empty == False:
        condition_accent = gender_chaos.name.notna() & (gender_chaos.name.str.encode(
            'ascii', errors='ignore') != gender_chaos.name.str.encode('ascii', errors='replace'))
        gender_chaos_accent = gender_chaos[condition_accent]
        if gender_chaos_accent.empty == False:
            gender_predict_accent = get_gender_from_name(
                gender_chaos_accent, type_name='accent', name_col='name')
            df_profile.loc[gender_chaos_accent.index,
                           'gender'] = gender_predict_accent

        condition_no_accent = gender_chaos.name.notna() & (gender_chaos.name.str.encode(
            'ascii', errors='ignore') == gender_chaos.name.str.encode('ascii', errors='replace'))
        gender_chaos_no_accent = gender_chaos[condition_no_accent]
        if gender_chaos_no_accent.empty == False:
            gender_predict_no_accent = get_gender_from_name(
                gender_chaos_no_accent, type_name='no_accent', name_col='name')
            df_profile.loc[gender_chaos_no_accent.index,
                           'gender'] = gender_predict_no_accent

    # Fillna gender
    condition_accent = df_profile.gender.isna() & df_profile.name.notna() & (df_profile.name.str.encode(
        'ascii', errors='ignore') != df_profile.name.str.encode('ascii', errors='replace'))
    df_profile_accent = df_profile[condition_accent]
    if df_profile_accent.empty == False:
        gender_predict_accent = get_gender_from_name(
            df_profile_accent, type_name='accent', name_col='name')

        df_profile.loc[df_profile_accent.index,
                       'gender'] = gender_predict_accent
        df_profile.loc[df_profile_accent.index,
                       'source_gender'] = 'name2gender'

    condition_no_accent = df_profile.gender.isna() & df_profile.name.notna()
    df_profile_no_accent = df_profile[condition_no_accent]
    if df_profile_no_accent.empty == False:
        gender_predict_no_accent = get_gender_from_name(
            df_profile_no_accent, type_name='no_accent', name_col='name')

        df_profile.loc[df_profile_no_accent.index,
                       'gender'] = gender_predict_no_accent
        df_profile.loc[df_profile_no_accent.index,
                       'source_gender'] = 'name2gender'

    # Save data
    print('Saving data...')
#     subprocess.call(f'hdfs dfs rm -R {tmp_path}', shell=True) # this use when run 2nd
    df_profile.to_parquet(tmp_path, index=False, filesystem=hdfs)

    stop = time()

    print(f'Time for fillna gender: {int(stop - start)}s')


def fillna_pipeline(id_cttv, input_path, tmp_path):

    # load data
    df_profile = pd.read_parquet(
        input_path, filesystem=hdfs).drop('part', axis=1)

    # fill ip location
    if id_cttv == 'user_id_fplay' or id_cttv == 'vne_id_fo':
        df_profile = fill_location(df_profile, id_cttv)

    # fillna email
    df_profile = fill_email(df_profile)

    # fillna gender
    fill_gender(df_profile, id_cttv, tmp_path)


def generate_profile_with_fid(spark, date, f_group, key_map):

    mapping_fids = spark.read.parquet(
        PREPROCESS_PATH + '/mapping_fids.parquet/d={}'.format(date.strftime('%F')))
    profile_non_fid = spark.read.parquet(
        f'{UNIFY_PATH}/{f_group}.parquet/d={date.strftime("%F")}')
    profile_non_fid = (profile_non_fid.withColumn('phone', F.when(F.col('is_phone_valid') != True, lit(None)).otherwise(F.col('phone')))
                                      .withColumn('email', F.when(F.col('is_email_valid') != True, lit(None)).otherwise(F.col('email')))
                       )

    profile = (profile_non_fid.drop('phone_raw', 'is_phone_valid', 'email_raw', 'is_email_valid', 'is_full_name', 'customer_type_detail')
               .filter(F.col(key_map).isNotNull())
               .dropDuplicates()
               .join(mapping_fids.filter(F.col(key_map).isNotNull()).dropDuplicates(), on=key_map, how='inner'))

    return profile


def generate_total_id(spark, date):

    start = time()

    fids = spark.read.parquet(FID_PATH + '/fid_core_score.parquet/d={}'.format(date.strftime("%F"))).filter(
        F.col('fid_score') >= 0.1).filter(F.col('phone_score') >= 0.1).select('fid', 'fid_score').dropDuplicates()

    fid_fo = spark.read.parquet(
        FID_PATH + '/fid_fo.parquet/d={}'.format(date.strftime("%F")))
    fid_fo = fid_fo.join(fids.select('fid'), on='fid', how='inner').select(
        'fid', 'vne_id_fo').dropDuplicates()

    fid_fplay = spark.read.parquet(
        FID_PATH + '/fid_fplay.parquet/d={}'.format(date.strftime("%F")))
    fid_fplay = fid_fplay.join(fids.select('fid'), on='fid', how='inner').select(
        'fid', 'user_id_fplay').dropDuplicates()

    fid_fshop = spark.read.parquet(
        FID_PATH + '/fid_fshop.parquet/d={}'.format(date.strftime("%F")))
    fid_fshop = fid_fshop.join(fids.select('fid'), on='fid', how='inner').select(
        'fid', 'cardcode_fshop').dropDuplicates()
    fid_fshop = fid_fshop.filter(~F.col('cardcode_fshop').rlike(
        '-')).withColumn('cardcode_fshop', F.col('cardcode_fshop').cast(IntegerType()))

    fid_longchau = spark.read.parquet(
        FID_PATH + '/fid_longchau.parquet/d={}'.format(date.strftime("%F")))
    fid_longchau = fid_longchau.join(fids.select('fid'), on='fid', how='inner').select(
        'fid', 'cardcode_longchau').dropDuplicates()
    fid_longchau = fid_longchau.filter(~F.col('cardcode_longchau').rlike(
        '-')).withColumn('cardcode_longchau', F.col('cardcode_longchau').cast(IntegerType()))

    fid_sendo = spark.read.parquet(
        FID_PATH + '/fid_sendo.parquet/d={}'.format(date.strftime("%F")))
    fid_sendo = fid_sendo.join(fids.select('fid'), on='fid', how='inner').select(
        'fid', 'id_sendo', 'phone')
    fid_sendo = (fid_sendo.withColumn('id_phone_sendo', F.when((F.col('id_sendo').isNotNull()) & (F.col('phone').isNotNull()), F.concat_ws('-', F.col('id_sendo'), F.col('phone')))
                                      .when((F.col('id_sendo').isNotNull()) & (F.col('phone').isNull()), F.col('id_sendo')))
                 .select('fid', 'id_phone_sendo')
                 .dropDuplicates())

    fid_ftel = spark.read.parquet(
        FID_PATH + '/fid_ftel.parquet/d={}'.format(date.strftime("%F")))
    fid_ftel = fid_ftel.join(fids.select('fid'), on='fid', how='inner').select(
        'fid', 'contract_ftel', 'phone')
    fid_ftel = (fid_ftel.withColumn('contract_phone_ftel', F.when((F.col('contract_ftel').isNotNull()) & (F.col('phone').isNotNull()), F.concat_ws('-', F.col('contract_ftel'), F.col('phone')))
                                    .when((F.col('contract_ftel').isNotNull()) & (F.col('phone').isNull()), F.col('contract_ftel')))
                        .select('fid', 'contract_phone_ftel')
                        .dropDuplicates())

    mapping_fids = (fids.join(fid_fo, on='fid', how='left')
                        .join(fid_fplay, on='fid', how='left')
                        .join(fid_fshop, on='fid', how='left')
                        .join(fid_longchau, on='fid', how='left')
                        .join(fid_sendo, on='fid', how='left')
                        .join(fid_ftel, on='fid', how='left')
                    )

    (mapping_fids.withColumn('d', lit(date.strftime('%F')))
                 .repartition(1)
                 .write.partitionBy("d").mode("overwrite").option("partitionOverwriteMode", "dynamic")
                 .parquet(PREPROCESS_PATH + '/mapping_fids.parquet'))

    stop = time()

    print(f'Time for mapping_fids: {int(stop - start)}s')


def post_processing(spark, date, cttv, tmp_path, save_path):

    start = time()

    df_profile = spark.read.parquet(tmp_path)

    # Fillna customer_type
    df_profile = (df_profile.withColumn('customer_type', F.when(F.col('name').isNotNull() & F.col('customer_type').isNull(), lit('Ca nhan'))
                                        .otherwise(F.col('customer_type')))
                  .withColumn('source_customer_type', F.when(F.col('customer_type').isNotNull() & F.col('source_customer_type').isNull(), lit(cttv))
                              .otherwise(F.col('source_customer_type')))
                  .withColumn('gender', F.when(F.col('customer_type') != 'Ca nhan', lit(None)).otherwise(F.col('gender')))
                  .withColumn('source_gender', F.when(F.col('customer_type') != 'Ca nhan', lit(None)).otherwise(F.col('source_gender')))
                  )

    # source Phone(only for FTEL)
    if cttv == 'FTEL':
        latest_file = sorted([f.path for f in hdfs.get_file_info(
            fs.FileSelector(RAW_PROFILE_PATH + '/ftel.parquet'))])[-1]
        ftel_source = spark.read.parquet(latest_file).withColumn('contract_phone_ftel', F.concat(
            F.col('contract_ftel'), lit('-'), F.col('phone'))).select('contract_phone_ftel', 'source')

        df_profile = (df_profile.join(ftel_source.filter(F.col('source') == 'demo').select('contract_phone_ftel').withColumn('is_demo', lit(True)), on='contract_phone_ftel', how='left')
                      .withColumn('source_phone', F.when((F.col('source_phone') == 'FTEL') & (F.col('is_demo') == True), lit('FTEL from demo'))
                                  .when((F.col('source_phone') == 'FTEL') & (F.col('is_demo').isNull()), lit('FTEL from multi'))
                                  .otherwise(F.col('source_phone'))
                                  )
                      )
    # source Phone(only for SENDO)
    if cttv == 'SENDO':
        latest_file = sorted([f.path for f in hdfs.get_file_info(
            fs.FileSelector(RAW_PROFILE_PATH + '/sendo.parquet'))])[-1]
        sendo_source = spark.read.parquet(latest_file).withColumn('id_phone_sendo', F.concat(
            F.col('id_sendo'), lit('-'), F.col('phone'))).select('id_phone_sendo', 'source')

        df_profile = (df_profile.join(sendo_source.filter(F.col('source') == 'buyer').select('id_phone_sendo').withColumn('is_buyer', lit(True)), on='id_phone_sendo', how='left')
                      .withColumn('source_phone', F.when((F.col('source_phone') == 'SENDO') & (F.col('is_buyer') == True), lit('SENDO from buyer'))
                                  .when((F.col('source_phone') == 'SENDO') & (F.col('is_buyer').isNull()), lit('SENDO from receiver'))
                                  .otherwise(F.col('source_phone'))
                                  )
                      )

    # Fix address
    FILLNA_COLS = ['unit_address', 'ward', 'district', 'city']
    DICT_TRASH = {'': None, 'Nan': None, 'nan': None,
                  'None': None, 'none': None, 'Null': None, 'null': None, "''": None}

    df_profile = (df_profile.withColumn('address', F.concat_ws(', ', F.col('unit_address'), F.col('ward'), F.col('district'), F.col('city')))
                  .withColumn('address', F.regexp_replace(F.col('address'), '(?<![a-zA-Z0-9]),', ''))
                  .withColumn('address', F.regexp_replace(F.col('address'), '-(?![a-zA-Z0-9])', ''))
                  .withColumn('address', F.trim(F.col('address'))))

    df_profile = df_profile.replace(DICT_TRASH, subset=['address'])
    df_profile = (df_profile.withColumn('source_address', F.when(F.col('address').isNotNull() & F.col('source_address').isNull(), F.col('source_city'))
                                        .when(F.col('address').isNull(), lit(None))
                                        .otherwise(F.col('source_address'))))

    # Fillna address if not have all 'ward', 'district', 'city'
    df_profile = (df_profile.withColumn('num_filled_address', reduce(add, [F.when(F.col(c).isNotNull(), lit(1)).otherwise(lit(0))
                                                                           for c in ['ward', 'district', 'city']]))
                  .withColumn('address', F.when(F.col('num_filled_address') != 3, lit(None)).otherwise(F.col('address')))
                  .withColumn('source_address', F.when(F.col('num_filled_address') != 3, lit(None)).otherwise(F.col('source_address')))
                  .drop('num_filled_address')
                  .dropDuplicates()
                  )
    # Saving data
    (df_profile.withColumn('d', lit(date.strftime('%F')))
               .repartition(1)
               .write.partitionBy("d").mode("overwrite").option("partitionOverwriteMode", "dynamic")
               .parquet(save_path))

    stop = time()

    print(f'Time for postprocessing data: {int(stop - start)}')


def choose_profile_representative(spark, date):

    start = time()

    profile_columns = ['fid', 'phone', 'source_phone', 'email', 'source_email', 'name', 'source_name',
                       'gender', 'source_gender', 'birthday', 'source_birthday', 'customer_type', 'source_customer_type',
                       'address', 'source_address', 'unit_address', 'source_unit_address', 'ward', 'source_ward',
                       'district', 'source_district', 'city', 'source_city']

    profile_ftel = spark.read.parquet(FILLNA_PATH + '/profile_ftel.parquet/d={}'.format(
        date.strftime("%F"))).select(profile_columns).withColumn('rank', lit(1))
    profile_fshop = spark.read.parquet(FILLNA_PATH + '/profile_fshop.parquet/d={}'.format(
        date.strftime("%F"))).select(profile_columns).withColumn('rank', lit(2))
    profile_longchau = spark.read.parquet(FILLNA_PATH + '/profile_longchau.parquet/d={}'.format(
        date.strftime("%F"))).select(profile_columns).withColumn('rank', lit(3))
    profile_sendo = spark.read.parquet(FILLNA_PATH + '/profile_sendo.parquet/d={}'.format(
        date.strftime("%F"))).select(profile_columns).withColumn('rank', lit(4))
    profile_fo = spark.read.parquet(FILLNA_PATH + '/profile_fo.parquet/d={}'.format(
        date.strftime("%F"))).select(profile_columns).withColumn('rank', lit(5))
    profile_fplay = spark.read.parquet(FILLNA_PATH + '/profile_fplay.parquet/d={}'.format(
        date.strftime("%F"))).select(profile_columns).withColumn('rank', lit(6))

    profile_cdp_total = profile_ftel.unionByName(profile_fshop).unionByName(profile_longchau).unionByName(
        profile_sendo).unionByName(profile_fo).unionByName(profile_fplay).cache()
    print(f'Profile total: {profile_cdp_total.count()}')

    # 1. Choose presentative from phone, campaign recently

    latest_file = sorted([f.path for f in hdfs.get_file_info(
        fs.FileSelector(PHONE_CAMPAIGN_PATH))])[-1]
    phone_campaign = spark.read.parquet(latest_file)
    phone_campaign_active = (phone_campaign.filter(F.col('date').isNotNull())
                             .filter(F.col('normalized_phone').isNotNull())
                             .withColumn('datediff', F.datediff(lit(date.today()), F.col('date')))
                             .filter(F.col('datediff') <= 365)
                             .select('normalized_phone')
                             .withColumnRenamed('normalized_phone', 'phone')
                             .dropDuplicates()
                             .cache()
                             )
    print(f'Phone campaign active: {phone_campaign_active.count()}')

    profile_with_phone_from_campaign = profile_cdp_total.join(
        phone_campaign_active, on='phone', how='inner')
    profile_with_phone_from_campaign = (profile_with_phone_from_campaign.withColumn('num_info', reduce(add, [F.when(F.col(c).isNotNull(), lit(1)).otherwise(lit(0)) for c in profile_columns]))
                                                                        .withColumn('rnb', F.row_number().over(Window.partitionBy('fid').orderBy(F.desc('num_info'), F.asc('rank'))))
                                                                        .filter(F.col('rnb') == 1)
                                                                        .drop('rank', 'rnb', 'num_info')
                                                                        .cache())
    print(f'Profile from campaign: {profile_with_phone_from_campaign.count()}')

    # 1.1 Get profile remain
    profile_cdp_remain = profile_cdp_total.join(
        profile_with_phone_from_campaign, on='fid', how='leftanti').cache()
    print(f'Profile remain: {profile_cdp_remain.count()}')

    # 2. Last active + Active date

    phone_profile_ftel = spark.read.parquet(RAW_PROFILE_PATH + '/ftel.parquet/d={}'.format(
        date.strftime("%F"))).select('phone', 'create_date').withColumn('last_active', lit(None))
    phone_profile_fshop = spark.read.parquet(RAW_PROFILE_PATH + '/fshop.parquet/d={}'.format(
        date.strftime("%F"))).select('phone', 'create_date', 'last_active')
    phone_profile_fo = spark.read.parquet(RAW_PROFILE_PATH + '/fo.parquet/d={}'.format(
        date.strftime("%F"))).select('phone', 'create_date', 'last_active')
    phone_profile_fplay = spark.read.parquet(RAW_PROFILE_PATH + '/fplay.parquet/d={}'.format(
        date.strftime("%F"))).select('phone', 'create_date', 'last_active')
    phone_profile_longchau = spark.read.parquet(RAW_PROFILE_PATH + '/longchau.parquet/d={}'.format(
        date.strftime("%F"))).select('phone', 'create_date', 'last_active')
    phone_profile_sendo = spark.read.parquet(RAW_PROFILE_PATH + '/sendo.parquet/d={}'.format(
        date.strftime("%F"))).select('phone', 'last_active').withColumn('create_date', lit(None))

    phone_profile = phone_profile_ftel.unionByName(phone_profile_fshop).unionByName(phone_profile_fo).unionByName(
        phone_profile_fplay).unionByName(phone_profile_longchau).unionByName(phone_profile_sendo)
    phone_profile = phone_profile.filter(F.col('phone').isNotNull())
    phone_profile = (phone_profile.withColumn('datediff_create_date', F.datediff(lit(date.today()), F.col('create_date')))
                                  .withColumn('datediff_last_active', F.datediff(lit(date.today()), F.col('last_active')))
                     )

    # 2.1 Filter last active(1 year)
    phone_profile_active = phone_profile.filter(
        F.col('datediff_last_active') <= 365).select('phone').dropDuplicates().cache()
    print(f'Phone profile active: {phone_profile_active.count()}')

    profile_with_phone_from_active = profile_cdp_remain.join(
        phone_profile_active, on='phone', how='inner')
    profile_with_phone_from_active = (profile_with_phone_from_active.withColumn('num_info', reduce(add, [F.when(F.col(c).isNotNull(), lit(1)).otherwise(lit(0)) for c in profile_columns]))
                                      .withColumn('rnb', F.row_number().over(Window.partitionBy('fid').orderBy(F.desc('num_info'), F.asc('rank'))))
                                      .filter(F.col('rnb') == 1)
                                      .drop('rank', 'rnb', 'num_info')
                                      .cache()
                                      )
    print(
        f'Profile from last active: {profile_with_phone_from_active.count()}')

    # 2.1.1 Get profile remain
    profile_cdp_remain = profile_cdp_remain.join(
        profile_with_phone_from_active, on='fid', how='leftanti')
    print(f'Profile remain: {profile_cdp_remain.count()}')

    # 2.2 Filter create date(in 2 years)

    phone_profile_create = phone_profile.filter(
        F.col('datediff_create_date') <= 365*2).select('phone').dropDuplicates().cache()
    print(f'Phone profile create_date: {phone_profile_create.count()}')

    profile_with_phone_from_create = profile_cdp_remain.join(
        phone_profile_create, on='phone', how='inner')
    profile_with_phone_from_create = (profile_with_phone_from_create.withColumn('num_info', reduce(add, [F.when(F.col(c).isNotNull(), lit(1)).otherwise(lit(0)) for c in profile_columns]))
                                      .withColumn('rnb', F.row_number().over(Window.partitionBy('fid').orderBy(F.desc('num_info'), F.asc('rank'))))
                                      .filter(F.col('rnb') == 1)
                                      .drop('rank', 'rnb', 'num_info')
                                      .cache()
                                      )
    print(
        f'Profile from create date: {profile_with_phone_from_create.count()}')

    # 2.2.1 Get profile remain
    profile_cdp_remain = profile_cdp_remain.join(
        profile_with_phone_from_create, on='fid', how='leftanti')
    print(f'Profile remain: {profile_cdp_remain.count()}')

    # 3. Keep more information + rank cttv(drop duplicates)

    demo_columns = ['fid', 'phone', 'source_phone', 'email', 'source_email', 'name', 'source_name',
                    'gender', 'source_gender', 'birthday', 'source_birthday', 'customer_type', 'source_customer_type', 'rank']

    geo_columns = ['fid', 'address', 'source_address', 'unit_address', 'source_unit_address', 'ward', 'source_ward',
                   'district', 'source_district', 'city', 'source_city', 'rank']

    # cdp demo
    cdp_demo = profile_cdp_remain.select(*demo_columns).dropDuplicates()
    cdp_demo = (cdp_demo.withColumn('num_info', reduce(add, [F.when(F.col(c).isNotNull(), lit(1)).otherwise(lit(0)) for c in demo_columns]))
                .withColumn('rnb', F.row_number().over(Window.partitionBy('fid').orderBy(F.desc('num_info'), F.asc('rank'))))
                .filter(F.col('rnb') == 1)
                .drop('rank', 'rnb', 'num_info'))

    # cdp geo
    cdp_geo = profile_cdp_remain.select(*geo_columns).dropDuplicates()
    cdp_geo = (cdp_geo.withColumn('num_info', reduce(add, [F.when(F.col(c).isNotNull(), lit(1)).otherwise(lit(0)) for c in geo_columns]))
                      .withColumn('rnb', F.row_number().over(Window.partitionBy('fid').orderBy(F.desc('num_info'), F.asc('rank'))))
                      .filter(F.col('rnb') == 1)
                      .drop('rank', 'rnb', 'num_info'))

    profile_with_phone_from_demo_geo = cdp_demo.join(
        cdp_geo, on='fid', how='inner').cache()
    print(
        f'Profile from demo + geo: {profile_with_phone_from_demo_geo.count()}')

    print('Union profile...')
    profile_total = profile_with_phone_from_campaign.unionByName(profile_with_phone_from_active).unionByName(
        profile_with_phone_from_create).unionByName(profile_with_phone_from_demo_geo)
    print(f'Profile unique: {profile_total.count()}')

    # 4. Post process
    profile_total = (profile_total.withColumn('gender', F.when(F.col('customer_type') != 'Ca nhan', lit(None)).otherwise(F.col('gender')))
                                  .withColumn('source_gender', F.when(F.col('customer_type') != 'Ca nhan', lit(None)).otherwise(F.col('source_gender')))
                                  .withColumn('birthday', F.when(F.col('customer_type') != 'Ca nhan', lit(None)).otherwise(F.col('birthday')))
                                  .withColumn('source_birthday', F.when(F.col('customer_type') != 'Ca nhan', lit(None)).otherwise(F.col('source_birthday')))
                                  .withColumn('d', F.lit(date.strftime("%F")))
                     )

    (profile_total.repartition(1)
                  .write.partitionBy("d").mode("overwrite").option("partitionOverwriteMode", "dynamic")
                  .parquet(UNIQUE_PATH + '/profile_cdp.parquet'))

    stop = time()

    print(f'Time elapsed for calculate profile unique: {int(stop - start)}s')
