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
    "birthday": "datetime64", # timestamp
    "age": "int",
    "address": "string",
    "city": "string",
    "district": "string",
    "ward": "string",
    "street": "string",
    "source": "string",
}

def centralize_data_frt_fshop(date):
    start = time()
    try:
        path = "/bigdata/fdp/frt/data/posdata/ict/pos_ocrd/"
        files = [
            path + file
            for file in os.listdir(path)
            #     if datetime.strptime(file.split('.')[0].split('_')[-1], '%Y-%m') >= start
        ]

        df_profile_list = []
        for file in sorted(files):
            print(file)
            df = pd.read_parquet(
                file,
                columns=[
                    "CardCode",
                    "Phone",
                    "Email",
                    "CardName",
                    "Gender",
                    "CustomerType",
                    "Address",
                    "City",
                    "CreateDate",
                ],
            )
            df["CreateDate"] = pd.to_datetime(
                df["CreateDate"], errors="coerce"
            )
            df_profile_list.append(df)
        df_profile = pd.concat(df_profile_list, ignore_index=True)
        df_profile = df_profile.loc[
            df_profile.CreateDate < date
        ]

        df_profile = (
            df_profile.sort_values("CreateDate", ascending=False)
            .drop_duplicates("CardCode", keep="first")
            .drop("CreateDate", axis=1)
        )
        df_profile.columns = [
            "uid",
            "phone",
            "email",
            "name",
            "gender",
            "customer_type",
            "address",
            "city",
        ]
        
        for col, _type in ORDERED_COLS.items():
            if col not in df_profile.columns:
                df_profile[col] = None
            # df_profile[col] = df_profile[col].astype(_type)
        df_profile = df_profile[list(ORDERED_COLS.keys())]
        print(
            f'[INFO] run_date={datetime.now()}; dag_date={date.strftime("%F")}; count={df_profile.shape[0]}'
        )
    except:
        print(
            ">>> /bigdata/fdp/frt/data/posdata/ict/pos_ocrd/ HASN'T BEEN UPDATED..."
        )
        pdate = date - timedelta(days=1)
        df_profile = pd.read_parquet(
            ROOT_PATH + "/centralize/frt_fshop.parquet/d={}".format(pdate.strftime("%F")),
            filesystem=hdfs,
        )
    finally:
        subprocess.call(
            f'hdfs dfs -rm -R {ROOT_PATH}/centralize/frt_fshop.parquet/d={date.strftime("%F")}',
            shell=True,
        )
        df_profile['d'] = date.strftime('%F')
        df_profile.to_parquet(
            ROOT_PATH + "/centralize/frt_fshop.parquet",
            partition_cols="d",
            filesystem=hdfs,
            index=False,
        )

    stop = time()
    print(f"Time elapsed: {int(stop - start)}s")


if __name__ == "__main__":
    
    date = datetime.strptime(sys.argv[1], "%Y-%m-%d")
    print(date)

    centralize_data_frt_fshop(date)
