import pandas as pd
import os

sub_phone_path = os.path.join(os.path.dirname(__file__), os.pardir, 'data', 'mobi_head_code.parquet')
sub_telephone_path = os.path.join(os.path.dirname(__file__), os.pardir, 'data', 'tele_head_code.parquet')

###? PHONE EXTRACTION
sub_phone_full = pd.read_parquet(sub_phone_path)

sub_telephone_full = pd.read_parquet(sub_telephone_path)

SUB_PHONE_10NUM = sorted(sub_phone_full["NewSubPhone"].unique())
SUB_PHONE_11NUM = [
    x for x in sorted(sub_phone_full["OldSubPhone"].unique()) if len(x) == 4
]

STATIC_PHONE_HEAD = ["0218", "0219", "0210", "0211"]
SUB_TELEPHONE_10NUM = [
    x
    for x in sorted(sub_telephone_full["ma_vung_cu"].unique())
    if x not in STATIC_PHONE_HEAD
]
SUB_TELEPHONE_11NUM = sorted(sub_telephone_full["ma_vung_moi"].unique())

DICT_4_SUB_PHONE = (
    sub_phone_full[sub_phone_full["OldSubPhone"].map(lambda x: len(x)) == 4]
    .set_index("OldSubPhone")
    .to_dict()["NewSubPhone"]
)

DICT_4_SUB_TELEPHONE = (
    sub_telephone_full[~sub_telephone_full["ma_vung_cu"].isin(STATIC_PHONE_HEAD)]
    .set_index("ma_vung_cu")
    .to_dict()["ma_vung_moi"]
)
