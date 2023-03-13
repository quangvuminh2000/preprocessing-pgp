
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


def parse_profile_from_transaction(spark, date):
    # Read transaction
    df_transaction = (
        spark.read.parquet(
            "/data/fpt/sendo/dwh/stag_sale_order.parquet/d={}".format(
                date.strftime("%F")
            )
        )
        .select(
            "buyer_id",
            "buyer_name",
            "buyer_phone",
            "buyer_email",
            "buyer_address",
            "buyer_district",
            "buyer_region",
            "order_time",
            "pod_time",
            "canceled_time",
            "receiver_name",
            "shipping_contact_phone",
            "ship_to_address",
            "ship_to_ward",
            "ship_to_ward_id",
            "ship_to_district",
            "ship_to_region",
        )
        .dropDuplicates()
    )

    df_location = (
        spark.read.parquet(
            ROOT_PATH + "/centralize/sendo_dictionary_location.parquet"
        )
        .select("place_id", "place_name")
        .withColumnRenamed("place_id", "ship_to_ward_id")
        .dropDuplicates()
    )

    # ward error
    df_transaction = df_transaction.withColumn(
        "noise_ship_ward",
        F.col("ship_to_ward").cast(IntegerType()).isNotNull(),
    )
    #     noise_ship_ward = df_transaction.select('ship_to_ward').cast('int').isNotNull()
    df_transaction1 = df_transaction.filter(
        ~F.col("noise_ship_ward")
    ).drop(
        "ship_to_ward_id", "noise_ship_ward"
    )  # not noise
    df_transaction2 = df_transaction.filter(
        F.col("noise_ship_ward")
    ).drop(
        "ship_to_ward", "noise_ship_ward"
    )  # noise

    df_transaction2 = (
        df_transaction2.join(
            df_location, on="ship_to_ward_id", how="left"
        )
        .withColumnRenamed("place_name", "ship_to_ward")
        .withColumn("buyer_address", F.lit(None))
        .withColumn("ship_to_address", F.lit(None))
        .drop("ship_to_ward_id")
    )

    df_transaction = df_transaction1.unionByName(
        df_transaction2
    )

    # noise name
    df_transaction = (
        df_transaction.withColumn(
            "buyer_name", F.initcap(F.trim(F.col("buyer_name")))
        )
        .withColumn(
            "buyer_name",
            F.when(
                F.col("buyer_name").isin(["Người Mua Pc3"]), F.lit(None)
            ).otherwise(F.col("buyer_name")),
        )
        .withColumn("receiver_name", F.initcap(F.trim(F.col("receiver_name"))))
        .withColumn(
            "receiver_name",
            F.when(
                F.col("receiver_name").isin(
                    ["Người Nhận Pc3", "Số Ngoài Danh Bạ"]
                ),
                F.lit(None),
            ).otherwise(F.col("receiver_name")),
        )
        .withColumn(
            "receiver_name",
            F.when(
                F.col("receiver_name").isNull(), F.col("buyer_name")
            ).otherwise(F.col("receiver_name")),
        )
    )

    # split: buyer, receiver
    buyer = df_transaction.select(
        "buyer_id",
        "buyer_phone",
        "buyer_email",
        "buyer_name",
        "buyer_address",
        "buyer_district",
        "buyer_region",
        "order_time",
        "pod_time",
    ).withColumn("source", F.lit("buyer"))

    receiver = df_transaction.select(
        "buyer_id",
        "shipping_contact_phone",
        "buyer_email",
        "receiver_name",
        "ship_to_address",
        "ship_to_ward",
        "ship_to_district",
        "ship_to_region",
        "order_time",
        "pod_time",
    ).withColumn("source", F.lit("receiver"))

    # rename
    buyer = rename_columns(
        buyer,
        {
            "buyer_id": "id",
            "buyer_name": "name",
            "buyer_phone": "phone",
            "buyer_email": "email",
            "buyer_address": "address",
            "buyer_district": "district",
            "buyer_region": "region",
        },
    )

    receiver = rename_columns(
        receiver,
        {
            "buyer_id": "id",
            "receiver_name": "name",
            "shipping_contact_phone": "phone",
            "buyer_email": "email",
            "ship_to_address": "address",
            "ship_to_ward": "ward",
            "ship_to_district": "district",
            "ship_to_region": "region",
        },
    )

    # address: strip, title
    buyer = (
        buyer.withColumn("ward", F.lit(None))
        .withColumn("address", F.initcap(F.trim(F.col("address"))))
        .withColumn(
            "address",
            F.when(
                F.col("address").isin(["", None, "none", "None"]), F.lit(None)
            ).otherwise(F.col("address")),
        )
        .withColumn(
            "flag_address",
            F.when(F.col("address").isNotNull(), F.lit(True)).when(
                F.col("address").isNull(), F.lit(False)
            ),
        )
    )
    receiver = (
        receiver.withColumn("address", F.initcap(F.trim(F.col("address"))))
        .withColumn(
            "address",
            F.when(
                F.col("address").isin(["", None, "none", "None"]), F.lit(None)
            ).otherwise(F.col("address")),
        )
        .withColumn(
            "flag_address",
            F.when(F.col("address").isNotNull(), F.lit(True)).when(
                F.col("address").isNull(), F.lit(False)
            ),
        )
    )

    # sort, drop_dup - keep the last record
    buyer = (
        buyer.withColumn(
            "rnb",
            F.row_number().over(
                Window.partitionBy("id", "phone").orderBy(
                    F.desc("flag_address"), F.desc("order_time")
                )
            ),
        )
        .filter(F.col("rnb") == 1)
        .drop("rnb")
    )

    receiver = (
        receiver.withColumn(
            "rnb",
            F.row_number().over(
                Window.partitionBy("id", "phone").orderBy(
                    F.desc("flag_address"), F.desc("order_time")
                )
            ),
        )
        .filter(F.col("rnb") == 1)
        .drop("rnb")
    )

    profile = buyer.unionByName(receiver)

    # unidecode address
    # profile = remove_accents(profile, "address")
    # profile = remove_accents(profile, "ward")
    # profile = remove_accents(profile, "district")
    # profile = remove_accents(profile, "region")

    # get columns
    profile = profile.select(
        "id",
        "phone",
        "email",
        "name",
        "address",
        "ward",
        "district",
        "region",
        "source",
    )
    profile = (
        profile.withColumn("customer_type", F.lit("Ca nhan"))
        .withColumn(
            "customer_type",
            F.when(
                F.lower(F.col("name")).rlike("cong ty|cty|ctcp"),
                F.lit("Cong ty"),
            ).otherwise(F.col("customer_type")),
        )
        .withColumn(
            "rnb",
            F.row_number().over(
                Window.partitionBy("id", "phone").orderBy(F.asc("source"))
            ),
        )
        .filter(F.col("rnb") == 1)
        .drop("rnb")
    )

    profile = rename_columns(
        profile,
        {
            "id": "uid",
            "region": "city",
        },
    )
    
    for col, _type in ORDERED_COLS.items():
        if col not in profile.columns:
            profile = profile.withColumn(col, F.lit(None).cast(_type))
        else:
            profile = profile.withColumn(col, F.col(col).cast(_type))
    profile = profile.select(list(ORDERED_COLS.keys()))

    return profile


def centralize_data_sendo_sendo(spark, date):
    start = time()
    try:
        # Get yesterday profile.
        pdate = date - timedelta(days=1)
        raw_yesterday = spark.read.parquet(
            ROOT_PATH + "/centralize/sendo_sendo.parquet/d={}".format(pdate.strftime("%F"))
        )

        raw_yesterday_buyer = raw_yesterday.filter(
            F.col("source") == "buyer"
        )
        raw_yesterday_receiver = raw_yesterday.filter(
            F.col("source") == "receiver"
        )

        # Get today profile from transaction.
        root = parse_profile_from_transaction(spark, pdate)
        root_buyer = root.filter(F.col("source") == "buyer")
        root_receiver = root.filter(F.col("source") == "receiver")

        # Append yesterday buyer to today buyer and
        # drop duplicates to keep the lastest
        # record for each frame.

        df_status = spark.read.parquet(ROOT_PATH + '/get_active_status/sendo_sendo.parquet/d={}'.format(date.strftime('%F')))
        root_buyer = (
            root_buyer.unionByName(raw_yesterday_buyer)
            .join(df_status, on='uid', how='left')
            .withColumn(
                "rnb",
                F.row_number().over(
                    Window.partitionBy("uid", "phone").orderBy(
                        F.desc("last_active")
                    )
                ),
            )
            .filter(F.col("rnb") == 1)
            .drop("rnb")
        )
        root_receiver = (
            root_receiver.unionByName(raw_yesterday_receiver)
            .join(df_status, on='uid', how='left')
            .withColumn(
                "rnb",
                F.row_number().over(
                    Window.partitionBy("uid", "phone").orderBy(
                        F.desc("last_active")
                    )
                ),
            )
            .filter(F.col("rnb") == 1)
            .drop("rnb")
        )

        # Why prioritize buyer?
        root = root_buyer.unionByName(root_receiver)

        # Handle cases when today's receiver was yesterday's buyer.
        buyer = (
            root.filter(F.col("source") == "buyer")
            .select("uid", "phone")
            .withColumn("is_buyer", F.lit(True))
        )

        raw = (
            root.withColumn(
                "rnb",
                F.row_number().over(
                    Window.partitionBy("uid", "phone").orderBy(
                        F.asc("source")
                    )
                ),  # Why prioritize buyer?
            )
            .filter(F.col("rnb") == 1)
            .drop("rnb")
            .join(buyer, on=["uid", "phone"], how="left")
        )

        raw = raw.withColumn(
            "source",
            F.when(F.col("is_buyer"), F.lit("buyer")).otherwise(
                F.col("source")
            ),
        )

        df_profile = (
            raw
            .withColumn(
                "rnb",
                F.row_number().over(
                    Window.partitionBy("uid").orderBy(
                        F.asc("first_active"), F.desc("last_active")
                    )
                ),
            )
            .filter(F.col("rnb") == 1)
            .drop("rnb")
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
            ">>> /data/fpt/sendo/dwh/stag_sale_order.parquet HASN'T BEEN UPDATED..."
        )
        pdate = date - timedelta(days=1)
        df_profile = spark.read.parquet(
            ROOT_PATH + "/centralize/sendo_sendo.parquet/d={}".format(pdate.strftime("%F"))
        )
    finally:
        (
            df_profile.withColumn("d", F.lit(date.strftime("%F")))
            .repartition(1)
            .write.partitionBy("d")
            .mode("overwrite")
            .option("partitionOverwriteMode", "dynamic")
            .parquet(ROOT_PATH + "/centralize/sendo_sendo.parquet")
        )

    stop = time()
    print(f"Time elapsed: {int(stop - start)}s")


##############################

if __name__ == "__main__":
    date = datetime.strptime(sys.argv[1], "%Y-%m-%d")
    print(date)

    spark = initial_spark(mode="yarn", driver_memory="10g", max_worker=5)
    centralize_data_sendo_sendo(spark, date)
    spark.stop()
