
import sys

sys.path.append("/bigdata/fdp/cdp/cdp_pages/scripts_hdfs/modules/")
# from preprocessing_pgp.card.validation import verify_card
from utils import *

cols_home = [
    "customer_name",
    "customer_phone",
    "customer_address",
    "card_id",
    "create_time",
]
cols_fe = [
    "customer_name",
    "customer_phone",
    "customer_address",
    "customer_card_id",
    "create_time",
]
cols_mirae_onl = ["full_name", "phone", "address", "card_id", "create_time"]
cols_mirae_off = [
    "first_name",
    "middle_name",
    "last_name",
    "customer_phone",
    "address",
    "birthday",
    "card_id",
    "gender",
    "create_time",
]
path_homcre = "/data/fpt/frt/fshop/dwh/fact_homecredit_installment.parquet/"
path_fecre = "/data/fpt/frt/fshop/dwh/fact_fe_credit_installment.parquet/"
path_miraeonl = (
    "/data/fpt/frt/fshop/dwh/fact_mirae_credit_online_installment.parquet/"
)
path_miraeoff = (
    "/data/fpt/frt/fshop/dwh/fact_mirae_credit_offline_installment.parquet/"
)
infor_cols = [
    "card_id",
    "phone",
    "name",
    "gender",
    "birthday",
    "address",
    "active_date",
    "last_active",
]

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

ROOT_PATH = "/data/fpt/ftel/cads/dep_solution/sa/cdp/core_profile"

def gender(gender):
    if (gender == "0") or (gender == "2"):
        gender = "Male"
    if (gender == "1") or (gender == "3"):
        gender = "Female"
    return gender


def birthday(year):
    if (year[0] == "0") or (year[0] == "1"):
        year = "19" + year[1:]
    if (year[0] == "2") or (year[0] == " 3"):
        year = "20" + year[1:]
    return year


listt = ["090", "091", "092", "230", "231", "245", "280", "281", "285"]


def cmnd(cmnd):
    if cmnd[:3] in listt:
        c = cmnd[:3]
    else:
        c = cmnd[:2]
    return c


def read_file_exists(path, date, list_cols, string):
    try:
        print(f"Having file {string}")
        read_file = pd.read_parquet(
            path + f"d={date}", columns=list_cols, filesystem=hdfs
        )
    except:
        print(f"File {string} is not exists")
        read_file = pd.read_parquet(columns=list_cols)
    return read_file


def parse_gender_birthday_from_cardid(data):
    # Part 1: cccd
    # ----->> Province code  <<-----
    # # ----->> Gender <<-----
    data["gender_new"] = data["card_id"]
    data.loc[
        (data["gender_new"].str.len() == 12) & (data["is_valid_cardid"]),
        "gender_new",
    ] = (
        data.loc[
            (data["gender_new"].str.len() == 12) & (data["is_valid_cardid"]),
            "gender_new",
        ]
        .str[3]
        .apply(lambda x: gender(x))
    )
    # # ----->> Birthday <<-----
    data["birthday_new"] = data["card_id"]
    data.loc[
        (data["birthday_new"].str.len() == 12) & (data["is_valid_cardid"]),
        "birthday_new",
    ] = (
        data.loc[
            (data["birthday_new"].str.len() == 12) & (data["is_valid_cardid"]),
            "birthday_new",
        ]
        .str[3:6]
        .apply(lambda x: birthday(x))
    )
    # *********************************************************************************************************************************************
    #
    # Part 2: cmnd
    # ----->> Gender <<-----
    data["birthday_new"] = data["birthday_new"].where(
        data["birthday_new"].str.len() == 4
    )
    # ----->> Birthday <<-----
    data["gender_new"] = data["gender_new"].where(
        data["gender_new"].str.len() != 9
    )
    # *********************************************************************************************************************************************
    #
    # Part 3: Driver license
    # # ----->> Gender <<-----
    data.loc[
        (data["gender_new"].str.len() == 12)
        & (data["is_valid_driver_license"]),
        "gender_new",
    ] = (
        data.loc[
            (data["gender_new"].str.len() == 12)
            & (data["is_valid_driver_license"]),
            "gender_new",
        ]
        .str[2]
        .apply(lambda x: gender(x))
    )
    return data


def save_data(data, date_segment, path):
    subprocess.call([f"hdfs dfs -rm -R {path}/d={date_segment}"], shell=True)
    data["d"] = date_segment
    data.to_parquet(
        f"{path}", partition_cols="d", index=False, filesystem=hdfs
    )


def centralize_data_frt_credit(date):
    try:
        ## Read file Homecre/ Fecre/ Miraeonl/Mỉaeoff
        data_homecre = read_file_exists(
            path_homcre, date, cols_home, "home_cre"
        )
        data_homecre = data_homecre.rename(
            columns={
                "customer_name": "name",
                "customer_phone": "phone",
                "customer_address": "address",
            }
        )

        data_fecre = read_file_exists(path_fecre, date, cols_fe, "fe_cre")
        data_fecre = data_fecre.rename(
            columns={
                "customer_name": "name",
                "customer_phone": "phone",
                "customer_address": "address",
                "customer_card_id": "card_id",
            }
        )

        data_miraeonl = read_file_exists(
            path_miraeonl, date, cols_mirae_onl, "mirae_onl"
        )
        data_miraeonl = data_miraeonl.rename(columns={"full_name": "name"})

        data_miraeoff = read_file_exists(
            path_miraeoff, date, cols_mirae_off, "mirae_off"
        )
        data_miraeoff["name"] = (
            data_miraeoff["first_name"]
            + " "
            + data_miraeoff["middle_name"]
            + " "
            + data_miraeoff["last_name"]
        )
        data_miraeoff["birthday"] = data_miraeoff["birthday"].dt.date
        data_miraeoff.rename(columns={"customer_phone": "phone"}, inplace=True)
        data_miraeoff.drop(
            columns=["first_name", "middle_name", "last_name"], inplace=True
        )
        data_credit = pd.concat(
            [data_homecre, data_fecre, data_miraeonl, data_miraeoff]
        ).reset_index(drop=True)

        # ## Verified Card id
        # verified_data = verify_card(data_credit, card_col="card_id")
        # data_credit["clean_card_id"] = verified_data["clean_card_id"].copy()
        # data_credit["is_valid_cardid"] = verified_data["is_personal_id"].copy()
        # data_credit["is_valid_driver_license"] = verified_data[
        #     "is_driver_license"
        # ].copy()
        # data_credit.drop(columns="card_id", inplace=True)
        # data_credit.rename(columns={"clean_card_id": "card_id"}, inplace=True)

        # ## Parse Gender/Birthday from card_id
        # data_credit_new = parse_gender_birthday_from_cardid(data_credit)
        # data_credit_new.loc[
        #     data_credit_new["gender"].isnull(), "gender"
        # ] = data_credit_new["gender_new"]
        # data_credit_new["gender"] = (
        #     data_credit_new["gender"]
        #     .map({"Male": "M", "Female": "F"})
        #     .fillna("-1")
        # )

        # data_credit_new.loc[
        #     data_credit_new["birthday"].isnull(), "birthday"
        # ] = data_credit_new["birthday_new"]
        # data_credit_new.loc[
        #     data_credit_new["birthday"].notna(), "birthday"
        # ] = data_credit_new["birthday"].astype(str)
        # data_credit_new.loc[
        #     data_credit_new["card_id"].notna(), "customer_type"
        # ] = "Ca nhan"
        # data_credit_new.loc[
        #     data_credit_new["card_id"].isnull(), "customer_type"
        # ] = None

#         data_profile = (
#             data_credit_new.sort_values("create_time", ascending=True)
#             .drop_duplicates("card_id", keep="first")
#             .rename(columns={"create_time": "active_date"})
#         )
#         last_active = (
#             data_credit_new[["card_id", "create_time"]]
#             .sort_values("create_time", ascending=True)
#             .drop_duplicates("card_id", keep="last")
#             .rename(columns={"create_time": "last_active"})
#         )

#         data_profile = data_profile.merge(
#             last_active, on="card_id", how="left"
#         )
#         data_profile["active_date"] = data_profile["active_date"].dt.date
#         data_profile["last_active"] = data_profile["last_active"].dt.date
#         data_profile.drop(
#             columns=[
#                 "is_valid_cardid",
#                 "is_valid_driver_license",
#                 "gender_new",
#                 "birthday_new",
#             ],
#             inplace=True,
#         )
        data_profile["d"] = date
    except:
        data_profile = pd.DataFrame(columns=infor_cols)
    return data_profile


if __name__ == "__main__":
    
    TODAY = sys.argv[1]
    YESTERDAY = (pd.to_datetime(TODAY) - timedelta(days=1)).strftime('%F')
    raw_today = centralize_data_frt_credit(TODAY)
    raw_today = raw_today.rename(columns={'card_id': 'uid'})

    raw_yesterday = pd.read_parquet(
        f"{ROOT_PATH}/centralize/frt_credit.parquet/d={YESTERDAY}", filesystem=hdfs
    )
    print("File today:", len(raw_today))
    print("File newlest:", len(raw_yesterday))
    df_profile = pd.concat([raw_yesterday, raw_today])
    print("File concat:", len(df_profile))

    # raw_total_new = (
    #     raw_total.sort_values("active_date", ascending=True)
    #     .drop_duplicates("uid", keep="first")
    #     .drop(columns=["last_active"])
    # )
    # print("File final:", len(raw_total_new))

    df_profile = df_profile[
        [
            "uid",
            "phone",
            "name",
            "gender",
            "customer_type",
            "birthday",
            "address",
        ]
    ]
    
    for col, _type in ORDERED_COLS.items():
        if col not in df_profile.columns:
            df_profile[col] = None
        # df_profile[col] = df_profile[col].astype(_type)
    df_profile = df_profile[list(ORDERED_COLS.keys())]
    
    save_data(df_profile, TODAY, f"{ROOT_PATH}/centralize/frt_credit.parquet")
