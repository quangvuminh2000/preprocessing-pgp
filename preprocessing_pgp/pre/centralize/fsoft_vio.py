
import sys

sys.path.append("/bigdata/fdp/cdp/cdp_pages/scripts_hdfs/modules/")
from utils import *

DIM_COLS = [
    "id",
    "user_name",
    "phone",
    "email",
    "first_name",
    "last_name",
    "full_name",
    "birthday",
    "grade",
    "edited_by",
    "force_update",
    "user_type",
    "short_id",
    "is_active",
    "is_verified",
    "is_locked",
    "is_deleted",
    "is_disabled",
    "is_unlocked_national_round",
    "school_name",
    "class_name",
    "address",
    "province_name",
    "district_name",
    "version_record",
    "version_account",
    "roles",
    "round_ids",
    "browser",
    "lock_session_id",
    "session_util",
    "create_date",
    "update_date",
    "d",
]

ROUND_RESULT_COLS = [
    "round_id",
    "version",
    "attempt_count",
    "duration",
    "finish_time",
    "is_finished_round",
    "round_number",
    "school_year",
    "subject_class",
    "subject_class_slug",
    "subject_name",
    "subject_slug",
    "score",
    "start_time",
    "total_score",
    "short_id",
    "user_name",
    "user_first_name",
    "user_full_name",
    "user_last_name",
    "user_grade",
    "user_class_name",
    "user_school_name",
    "user_district_name",
    "user_province_name",
    "d",
]

hdfs_path_profile = (
    "/data/fpt/ftel/cads/dep_solution/sa/fsoft/vio/dim_user.parquet"
)
hdfs_path_history_profile = (
    "/data/fpt/ftel/cads/dep_solution/sa/fsoft/vio/dim_history_user.parquet"
)
hdfs_path_round = (
    "/data/fpt/ftel/cads/dep_solution/sa/fsoft/vio/ds_round_result.parquet"
)
hdfs_path_round_details = "/data/fpt/ftel/cads/dep_solution/sa/fsoft/vio/ds_round_result_details.parquet"

ROOT_PATH = "/data/fpt/ftel/cads/dep_solution/sa/cdp/core_profile"
DEMO_COLUMNS = [
    "user_name",
    "phone",
    "email",
    "full_name",
    "birthday",
    "user_type",
    "address",
    "province_name",
    "district_name",
    "create_date",
]

###########################################################################################################################

ORDERED_COLS = {
    "uid": "string",
    "phone": "string",
    "email": "string",
    "name": "string",
    "gender": "string",
    "customer_type": "string",
    "birthday": "timestamp",
    "age": "int",
    "address": "string",
    "city": "string",
    "district": "string",
    "ward": "string",
    "street": "string",
    "source": "string",
}

def migrate_data_fsoft_vio(date):
    import spark_sdk as ss

    spark = ss.PySpark(
        yarn=True, driver_memory="8G", num_executors=4, executor_memory="16G"
    ).spark
    
    # profile
    df_dim_user = (
        spark.sql("select * from fsoft_dwh_violympic.dim_user")
        .withColumn("d", F.lit(date.strftime('%F')))
        .select(DIM_COLS)
        .withColumn(
            "birthday", F.col("birthday").cast("string")
        )
        .withColumn("create_date", F.col("create_date").cast("string"))
        .withColumn("update_date", F.col("update_date").cast("string"))
    )

    # history profile
    df_dim_history_user = (
        spark.sql("select * from fsoft_dwh_violympic.dim_history_user")
        .withColumn("d", F.lit(date.strftime('%F')))
        .select(DIM_COLS)
        .withColumn(
            "birthday", F.col("birthday").cast("string")
        )
        .withColumn("create_date", F.col("create_date").cast("string"))
        .withColumn("update_date", F.col("update_date").cast("string"))
    )

    # round
    df_round_result = spark.sql(
        "select * from fsoft_dwh_violympic.ds_round_result"
    ).select(ROUND_RESULT_COLS)
    df_round_result_detail = spark.sql(
        "select * from fsoft_dwh_violympic.ds_round_result_detail"
    )
    
    # save data
    (
        df_dim_user.repartition(1)
        .write.partitionBy("d")
        .mode("overwrite")
        .option("partitionOverwriteMode", "dynamic")
        .parquet(hdfs_path_profile)
    )
    
    (
        df_dim_history_user.repartition(1)
        .write.partitionBy("d")
        .mode("overwrite")
        .option("partitionOverwriteMode", "dynamic")
        .parquet(hdfs_path_history_profile)
    )


    (
        df_round_result.repartition(1)
        .write.partitionBy("d")
        .mode("overwrite")
        .option("partitionOverwriteMode", "dynamic")
        .parquet(hdfs_path_round)
    )

    (
        df_round_result_detail.repartition(1)
        .write.partitionBy("d")
        .mode("overwrite")
        .option("partitionOverwriteMode", "dynamic")
        .parquet(hdfs_path_round_details)
    )


def centralize_data_fsoft_vio(spark, date):
    
    pdate = date - timedelta(days=1)
    try:
        df_profile = spark.read.parquet(hdfs_path_profile + '/d={}'.format(date.strftime('%F')))
        
        # new uid
        df_uid = (
            df_profile.select('user_name')
            .filter(F.col('user_name').isNotNull())
            .filter(F.trim('user_name') != '')
            .distinct()
        )
        df_old_uid = None
        max_old_uid = -1
        try:
            df_old_uid = spark.read.parquet(ROOT_PATH + '/centralize/fsoft_vio_uid_mapping.parquet')
            max_old_uid = df_old_uid.agg(F.max('uid')).collect()[0][0]
            df_uid = df_uid.join(
                df_old_uid.select('user_name'),
                on='user_name',
                how='left_anti'
            )
        except:
            pass
        df_uid = (df_uid
            .withColumn(
                'uid',
                F.row_number().over(Window.orderBy(F.asc('user_name'))) + max_old_uid
        ).withColumn('update_date', F.lit(date.strftime('%F'))))
        # if df_old_uid:
        #     df_uid = df_old_uid.unionByName(df_uid)
        (
            df_uid.select('uid', 'user_name', 'update_date').write.mode("append")
            .parquet(ROOT_PATH + "/centralize/fsoft_vio_uid_mapping.parquet")
        )
        
        # profile
        df_profile = (
            df_profile
            .select(DEMO_COLUMNS)
            .join(
                df_uid.select('uid', 'user_name'),
                on='user_name',
                how='left'
            )
            .filter(
                F.col('uid').isNotNull()
                & (
                    F.col('phone').isNotNull()
                    | F.col('email').isNotNull()
                )
            )
            .withColumn(
                'rnb',
                F.row_number().over(Window.partitionBy('uid', 'phone', 'email', 'full_name').orderBy(F.desc('create_date')))
            )
            .filter(F.col('rnb') == 1)
            .drop('rnb')
        )

        df_profile = (
            df_profile
            .withColumn(
                'customer_type',
                F.when(F.col('user_type').isin(["TEACHER", "STUDENT"]), F.lit('Ca nhan'))
                .otherwise(F.lit(None))
            )
        )

        df_profile = rename_columns(
            df_profile, {
                # "short_id": "uid",
                "full_name": "name",
                "province_name": "city",
                "district_name": "district",
            }
        )

        for col, _type in ORDERED_COLS.items():
            if col not in df_profile.columns:
                df_profile = df_profile.withColumn(col, F.lit(None).cast(_type))
            else:
                df_profile = df_profile.withColumn(col, F.col(col).cast(_type))
        df_profile = df_profile.select(list(ORDERED_COLS.keys()))
        
        # get data yesterday, concat and dropDuplicates
        try:
            df_profile_yesterday = spark.read.parquet(
                ROOT_PATH + "/centralize/fsoft_vio.parquet/d={}".format(pdate.strftime("%F"))
            )
            df_profile = (
                df_profile.withColumn("d", F.lit(date.strftime("%F")))
                .unionByName(df_profile_yesterday.withColumn("d", F.lit(pdate.strftime("%F"))))
                .withColumn(
                    'rnb',
                    F.row_number().over(Window.partitionBy('uid', 'phone', 'email', 'name').orderBy(F.desc('d')))
                )
                .filter(F.col('rnb') == 1)
                .drop('rnb')
            )
        except:
            print('Init date !')
            
        print(
                f'[INFO] run_date={datetime.now()}; dag_date={date.strftime("%F")}; count={df_profile.count()}'
            )
    except:
        print(
            ">>> /data/fpt/ftel/cads/dep_solution/sa/fsoft/vio/dim_user.parquet HASN'T BEEN UPDATED..."
        )
        raise # remove
        df_profile = spark.read.parquet(
            ROOT_PATH + "/centralize/fsoft_vio.parquet/d={}".format(pdate.strftime("%F"))
        )
    finally:
        (
            df_profile
            .withColumn("d", F.lit(date.strftime("%F")))
            .repartition(1)
            .write.partitionBy("d")
            .mode("overwrite")
            .option("partitionOverwriteMode", "dynamic")
            .parquet(ROOT_PATH + "/centralize/fsoft_vio.parquet")
        )

if __name__ == "__main__":
    date = datetime.strptime(sys.argv[1], "%Y-%m-%d")
    print(date)

    spark = initial_spark(mode="yarn", driver_memory="10g", max_worker=5)
    # migrate_data_fsoft_vio(date)
    centralize_data_fsoft_vio(spark, date)
    spark.stop()
