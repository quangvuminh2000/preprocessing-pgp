import sys

sys.path.append("/bigdata/fdp/cdp/cdp_pages/scripts_hdfs/modules/")
from utils import *

ROOT_PATH = "/data/fpt/ftel/cads/dep_solution/sa/cdp/core_profile"

PHONE_KEY = "mI3Q68rOdAhh5hekRGUzlw=="
EMAIL_KEY = "m2EzGDWUA5nj4e4b+5p48Q=="

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


def centralize_data_ftel_fplay(spark, date):
    start = time()
    try:
        df_profile = spark.sql(
            """SELECT user_id, id, name, create_time,
                fdecrypt(phone, '{}') as phone,
                fdecrypt(email, '{}') as email
                FROM ftel_dwh_fplay.dim_user where d = '{}'""".format(
                PHONE_KEY, EMAIL_KEY, date.strftime("%F")
            )
        )

        df_profile = (
            df_profile.withColumn(
                "rnb",
                F.row_number().over(
                    Window.partitionBy("user_id").orderBy(
                        F.desc("create_time")
                    )
                ),
            )
            .filter(F.col("rnb") == 1)
            .withColumnRenamed("user_id", "user_id_fplay")
            .withColumnRenamed("id", "user_id_private")
            .drop("rnb", "create_time")  # not use create_time in dim_user
        )
        
        # out mapping table fplay ids
        mapping_id_ftel_fplay = df_profile.select('user_id_fplay', 'user_id_private').dropDuplicates()
        df_profile = df_profile.withColumnRenamed("user_id_fplay", "uid")
        
        for col, _type in ORDERED_COLS.items():
            if col not in df_profile.columns:
                df_profile = df_profile.withColumn(col, F.lit(None).cast(_type))
            else:
                df_profile = df_profile.withColumn(col, F.col(col).cast(_type))
        df_profile = df_profile.select(list(ORDERED_COLS.keys()))

        print(
            f'[INFO] run_date={datetime.now()}; dag_date={date.strftime("%F")}; count={df_profile.count()}'
        )
    except:
        print(">>> dim_user HASN'T BEEN UPDATED...")
        pdate = date - timedelta(days=1)
        mapping_id_ftel_fplay = spark.read.parquet(
            ROOT_PATH + "/centralize/mapping_id_ftel_fplay.parquet/d={}".format(pdate.strftime("%F"))
        )
        df_profile = spark.read.parquet(
            ROOT_PATH + "/centralize/ftel_fplay.parquet/d={}".format(pdate.strftime("%F"))
        )
    finally:
        (
            mapping_id_ftel_fplay
            .withColumn("d", F.lit(date.strftime("%F")))
            .repartition(1)
            .write.partitionBy("d")
            .mode("overwrite")
            .option("partitionOverwriteMode", "dynamic")
            .parquet(ROOT_PATH + "/centralize/mapping_id_ftel_fplay.parquet")
        )
        (
            df_profile
            .withColumn("d", F.lit(date.strftime("%F")))
            .repartition(1)
            .write.partitionBy("d")
            .mode("overwrite")
            .option("partitionOverwriteMode", "dynamic")
            .parquet(ROOT_PATH + "/centralize/ftel_fplay.parquet")
        )

    stop = time()
    print(f"Time elapsed: {int(stop - start)}s")


if __name__ == "__main__":
    date = datetime.strptime(sys.argv[1], "%Y-%m-%d")
    print(date)

    spark = initial_spark(mode="yarn", driver_memory="10g", max_worker=5)
    centralize_data_ftel_fplay(spark, date)
    spark.stop()
