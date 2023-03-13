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


def centralize_data_frt_longchau(spark, date):
    start = time()
    try:
        df_profile = (
            spark.read.parquet(
                "file:///bigdata/fdp/frt/data/posdata/pharmacy/posthuoc_ocrd/*"
            )
            .select(
                "CardCode",
                "Phone",
                "Email",
                "CardName",
                "Gender",
                "CustomerType",
                "Address",
                "City",
                "CreateDate",
            )
            .withColumn(
                "rnb",
                F.row_number().over(
                    Window.partitionBy("CardCode").orderBy(
                        F.desc("CreateDate")
                    )
                ),
            )
            .filter(F.col("rnb") == 1)
            .drop("rnb", "CreateDate")  # not use CreateDate from ocrd
        )

        df_profile = rename_columns(
            df_profile,
            {
                "CardCode": "uid",
                "Phone": "phone",
                "Email": "email",
                "CardName": "name",
                "Gender": "gender",
                "CustomerType": "customer_type",
                "Address": "address",
                "City": "city",
            },
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
            ">>> /bigdata/fdp/frt/data/posdata/pharmacy/posthuoc_ocrd HASN'T BEEN UPDATED..."
        )
        pdate = date - timedelta(days=1)
        df_profile = spark.read.parquet(
            ROOT_PATH
            + "/centralize/frt_longchau.parquet/d={}".format(pdate.strftime("%F"))
        )
    finally:
        (
            df_profile
            .withColumn("d", F.lit(date.strftime("%F")))
            .repartition(1)
            .write.partitionBy("d")
            .mode("overwrite")
            .option("partitionOverwriteMode", "dynamic")
            .parquet(ROOT_PATH + "/centralize/frt_longchau.parquet")
        )

    stop = time()
    print(f"Time elapsed: {int(stop - start)}s")


if __name__ == "__main__":
    date = datetime.strptime(sys.argv[1], "%Y-%m-%d")
    print(date)

    spark = initial_spark()
    centralize_data_frt_longchau(spark, date)
    spark.stop()
