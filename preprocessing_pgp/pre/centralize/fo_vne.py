
import sys

sys.path.append("/bigdata/fdp/cdp/cdp_pages/scripts_hdfs/modules/")
from utils import *

ROOT_PATH = "/data/fpt/ftel/cads/dep_solution/sa/cdp/core_profile"

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


def centralize_data_fo_vne(spark, date):
    start = time()

    try:
        df_profile = spark.read.parquet(
            "/data/fpt/fo/dwh/stag_user_profile.parquet/d={}".format(
                date.strftime("%F")
            )
        ).select(
            "vne_id", "phone", "email", "name", "gender", "age", "address"
        )
        
        df_status = spark.read.parquet(ROOT_PATH + '/get_active_status/fo_vne.parquet/d={}'.format(date.strftime('%F'))).select('uid', 'first_active').withColumnRenamed('uid', 'vne_id')

        df_profile = (
            df_profile
            .join(df_status, on='vne_id', how='left')
        )

        df_profile = (
            df_profile.replace("", None, subset=df_profile.columns)
            .withColumn(
                "rnb",
                F.row_number().over(
                    Window.partitionBy("vne_id", "phone").orderBy(
                        F.desc("first_active")
                    )
                ),
            )
            .filter(F.col("rnb") == 1)
            .drop("rnb")
            .withColumnRenamed('vne_id', 'uid')
        )

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
        print(
            ">>> /data/fpt/fo/dwh/stag_user_profile.parquet HASN'T BEEN UPDATED..."
        )
        pdate = date - timedelta(days=1)
        df_profile = spark.read.parquet(
            ROOT_PATH + "/centralize/fo_vne.parquet/d={}".format(pdate.strftime("%F"))
        )
    finally:
        (
            df_profile
            .withColumn("d", F.lit(date.strftime("%F")))
            .repartition(1)
            .write.partitionBy("d")
            .mode("overwrite")
            .option("partitionOverwriteMode", "dynamic")
            .parquet(ROOT_PATH + "/centralize/fo_vne.parquet")
        )

    stop = time()
    print(f"Time elapsed: {int(stop - start)}s")


if __name__ == "__main__":
    date = datetime.strptime(sys.argv[1], "%Y-%m-%d")
    print(date)

    spark = initial_spark(mode="yarn", driver_memory="10g", max_worker=5)
    centralize_data_fo_vne(spark, date)
    spark.stop()
