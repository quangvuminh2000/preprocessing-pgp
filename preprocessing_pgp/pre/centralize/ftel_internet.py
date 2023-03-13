
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


def centralize_data_ftel_internet(spark, date):
    start = time()
    try:
        # Use demographic datasets for customer's basic info.
        pdate = date - timedelta(days=1)
        df_demo = (
            spark.read.parquet(
                "/data/fpt/ftel/cads/dep_solution/sa/ftel/internet/data/rst_demographics.parquet/d={}".format(
                    pdate.strftime("%F")
                )
            )
            .select(
                "contract",
                "phone",
                "email",
                "customer_name",
                "birthday",
                "address",
                "street",
                "district",
                "ward",
                "entity",
            )
            .filter(
                ~F.lower(F.col("customer_name")).rlike("test|demo|account")
            )  # remove contract test|demo|thu nghiem
            .withColumnRenamed("customer_name", "name")
            .withColumnRenamed("entity", "customer_type")
            .withColumn("source", F.lit("demo"))
        )
        duplicated_demo = (
            df_demo.groupBy("contract")
            .count()
            .filter(F.col("count") > 1)
            .select("contract")
            .distinct()
        )
        # Remove duplicated
        df_demo = df_demo.join(
            duplicated_demo, on="contract", how="left_anti"
        )

        # More phones from multiphone dataset.
        df_multi = (
            spark.read.parquet(
                "/data/fpt/ftel/cads/dep_solution/sa/ftel/internet/data/rst_multiphone.parquet/d={}".format(
                    pdate.strftime("%F")
                )
            )
            .select("contract", "phone", "description")
            .withColumnRenamed("description", "name")
            .filter(F.col("contract").isNotNull())
            .filter(
                ~F.lower(F.col("name")).rlike("test|demo|thu nghiem")
            )  # remove contract test|demo|thu nghiem
            .withColumn("source", F.lit("multi"))
        )
        # keys: ['contract', 'phone']
        # prioritize keys in demo, remove the same keys in multiphone
        df_multi_remain = df_multi.join(
            df_demo, on=["contract", "phone"], how="left_anti"
        )
        missing_cols = [
            c
            for c in df_demo.columns
            if c not in df_multi_remain.columns
        ]
        for c in missing_cols:
            df_multi_remain = df_multi_remain.withColumn(
                c, F.lit(None)
            )

        df_profile = df_demo.unionByName(df_multi_remain)

        # Load data paytv for package info.
        df_contract = (
            spark.read.parquet(
                "/data/fpt/ftel/cads/dep_solution/sa/ftel/internet/data/datapay.parquet/d={}".format(
                    date.strftime("%F")
                )
            )
            .withColumnRenamed("province", "city")
            .filter(
                F.col("current_net_package").isin(
                    [
                        "Giga",
                        "Sky",
                        "Meta",
                        "FTTH - Super30",
                        "FTTH - Super80",
                        "FTTH - Super100",
                        "FTTH - Super150",
                        "FTTH - Super250",
                        "FTTH - Super400",
                    ]
                )
            )
            .select('contract', 'city')
        )

        df_profile = (
            df_profile.join(df_contract, on="contract", how="inner")
            .dropDuplicates(subset=["contract", "phone"])
            .withColumnRenamed("contract", "uid")
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
        print(">>> demographics and multiphone HASN'T BEEN UPDATED...")
        pdate = date - timedelta(days=1)
        df_profile = spark.read.parquet(
            ROOT_PATH + "/centralize/ftel_internet.parquet/d={}".format(pdate.strftime("%F"))
        )
    finally:
        (
            df_profile.withColumn("d", F.lit(date.strftime("%F")))
            .repartition(1)
            .write.partitionBy("d")
            .mode("overwrite")
            .option("partitionOverwriteMode", "dynamic")
            .parquet(ROOT_PATH + "/centralize/ftel_internet.parquet")
        )

    stop = time()
    print(f"Time elapsed: {int(stop - start)}s")


if __name__ == "__main__":
    date = datetime.strptime(sys.argv[1], "%Y-%m-%d")
    print(date)

    spark = initial_spark(mode="yarn", driver_memory="10g", max_worker=5)
    centralize_data_ftel_internet(spark, date)
    spark.stop()
