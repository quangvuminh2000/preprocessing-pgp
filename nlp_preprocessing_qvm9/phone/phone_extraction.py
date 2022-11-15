import os
import subprocess
import multiprocessing as mp
from pyarrow import fs

import pandas as pd
import numpy as np
from unidecode import unidecode
from tqdm import tqdm


# ? ENVIRONMENT SETUP
os.environ["HADOOP_CONF_DIR"] = "/etc/hadoop/conf/"
os.environ["JAVA_HOME"] = "/usr/jdk64/jdk1.8.0_112"
os.environ["HADOOP_HOME"] = "/usr/hdp/3.1.0.0-78/hadoop"
os.environ["ARROW_LIBHDFS_DIR"] = "/usr/hdp/3.1.0.0-78/usr/lib/"
os.environ["CLASSPATH"] = subprocess.check_output(
    "$HADOOP_HOME/bin/hadoop classpath --glob", shell=True
).decode("utf-8")
hdfs = fs.HadoopFileSystem(host="hdfs://hdfs-cluster.datalake.bigdata.local", port=8020)

N_PROCESSES = 20

# ? DATA PREPARATION
subphone_vn = pd.read_parquet(
    "/data/fpt/ftel/cads/dep_solution/sa/cdp/data/vn_sub_phone.parquet", filesystem=hdfs
).set_index("PhoneVendor")
subphone_vn = "0" + subphone_vn.astype(str)
df_subtele_vn = pd.read_parquet(
    "/data/fpt/ftel/cads/dep_solution/sa/cdp/data/vn_sub_telephone.parquet",
    filesystem=hdfs,
)
SUB_PHONE_10NUM = sorted(subphone_vn["NewSubPhone"].unique())
SUB_PHONE_11NUM = [
    x for x in sorted(subphone_vn["OldSubPhone"].unique()) if len(x) == 4
]
SUB_TELEPHONE = sorted(df_subtele_vn["MaVung"].unique())
DICT_4SUBPHONE = (
    subphone_vn[subphone_vn["OldSubPhone"].map(lambda x: len(x)) == 4]
    .set_index("OldSubPhone")
    .to_dict()["NewSubPhone"]
)

# ? CHECK & EXTRACT FOR VALID PHONE
def extract_valid_phone(
    f_phones: pd.DataFrame, phone_col: str = "phone"
) -> pd.DataFrame:
    # Replace any characters that are not number with empty string.
    f_phones[phone_col] = (
        f_phones[phone_col]
        .str.replace(r"[^0-9]", "", regex=True)
        .str.strip()
        .str.replace("\s+", "", regex=True)
    )

    with mp.Pool(N_PROCESSES) as pool:
        f_phones["phone_length"] = tqdm(
            pool.imap(f_phones[phone_col].map(lambda x: len(str(x)))),
            total=f_phones.shape[0],
        )
    # Phone length validation: currently support phone number with length of 10 and 11.
    # Also, phone prefix has to be in the sub-phone dictionary.
    f_phones.loc[
        (
            (f_phones["phone_length"] == 10)
            & (f_phones[phone_col].str[:3].isin(SUB_PHONE_10NUM))
        ),
        "is_phone_valid",
    ] = True
    print(
        f"# OF PHONE 10 NUM VALID : {f_phones[f_phones['is_phone_valid']].shape[0]}",
        end="\n\n\n",
    )

    f_phones.loc[
        (
            (f_phones["phone_length"] == 11)
            & (f_phones[phone_col].str[:4].isin(SUB_PHONE_11NUM))
        ),
        "is_phone_valid",
    ] = True
    print(
        f"# OF PHONE 11 NUM VALID : {f_phones[(f_phones['is_phone_valid']) & (f_phones['phone_length'] == 11)].shape[0]}",
        end="\n\n\n",
    )

    f_phones = f_phones.reset_index(drop=True)

    # Correct phone numbers with old phone number format.
    f_phones.loc[
        (f_phones["phone_length"] == 11) & (f_phones["is_phone_valid"] == True),
        "phone_convert",
    ] = f_phones.loc[
        (f_phones["phone_length"] == 11) & (f_phones["is_phone_valid"] == True),
        phone_col,
    ].map(
        lambda x: DICT_4SUBPHONE[x[:4]] + x[4:] if (x[:4] in SUB_PHONE_11NUM) else None
    )

    print(f"# OF OLD PHONE CONVERTED : {f_phones['phone_convert'].notna().sum()}")

    # Check for tele-phone.
    f_phones.loc[
        (
            (f_phones["phone_length"] == 11)
            & (f_phones[phone_col].str[:3].isin(SUB_TELEPHONE))
        ),
        "is_phone_valid",
    ] = True

    print(
        f"# OF VALID TELEPHONE : {f_phones[(f_phones['phone_length'] == 11) & (f_phones[phone_col].str[:3].isin(SUB_TELEPHONE))]}",
        end="\n\n\n",
    )

    f_phones["is_phone_valid"] = f_phones["is_phone_valid"].fillna(False)
    f_phones = f_phones.drop("phone_length", axis=1)
    f_phones.loc[
        f_phones["is_phone_valid"] & f_phones["phone_convert"].isna(), "phone_convert"
    ] = f_phones[phone_col]

    print(
        f"# OF VALID PHONE : {f_phones[f_phones['is_phone_valid']].shape[0]}",
        end="\n\n",
    )
    print(
        f"# OF INVALID PHONE : {f_phones[~f_phones['is_phone_valid']].shape[0]}",
        end="\n\n",
    )

    print("Sample of invalid phones:", end="\n\n")
    print(f_phones[~f_phones["is_phone_valid"]].head(10))

    return f_phones
