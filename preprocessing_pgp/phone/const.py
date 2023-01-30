import pandas as pd
import os

mobi_phone_path = os.path.join(
    os.path.dirname(__file__),
    os.pardir,
    'data',
    'mobi_head_code.parquet'
)
telephone_path = os.path.join(
    os.path.dirname(__file__),
    os.pardir,
    'data',
    'tele_head_code.parquet'
)

# ? PHONE EXTRACTION
sub_mobi_phone = pd.read_parquet(mobi_phone_path).reset_index()

sub_telephone = pd.read_parquet(telephone_path).reset_index()

SUB_PHONE_10NUM = sorted(sub_mobi_phone["NewSubPhone"].unique())
SUB_PHONE_11NUM = [
    x for x in sorted(sub_mobi_phone["OldSubPhone"].unique()) if len(x) == 4
]

STATIC_PHONE_HEAD = ["0218", "0219", "0210", "0211"]
SUB_TELEPHONE_10NUM = [
    x
    for x in sorted(sub_telephone["ma_vung_cu"].unique())
    if x not in STATIC_PHONE_HEAD
]
SUB_TELEPHONE_11NUM = sorted(sub_telephone["ma_vung_moi"].unique())

DICT_4_SUB_PHONE = (
    sub_mobi_phone[sub_mobi_phone["OldSubPhone"].str.len() == 4]
    .set_index("OldSubPhone")
    .to_dict()["NewSubPhone"]
)

DICT_4_SUB_TELEPHONE = (
    sub_telephone[~sub_telephone["ma_vung_cu"].isin(
        STATIC_PHONE_HEAD)]
    .set_index("ma_vung_cu")
    .to_dict()["ma_vung_moi"]
)

# ? PHONE VENDOR
DICT_NEW_MOBI_PHONE_VENDOR = (
    sub_mobi_phone.set_index('NewSubPhone')
    .to_dict()['PhoneVendor']
)

DICT_NEW_TELEPHONE_VENDOR = (
    sub_telephone.set_index('ma_vung_moi')
    .to_dict()['tinh']
)

# ? PHONE LENGTH
PHONE_LENGTH = {
    "old_mobi": 11,
    "new_mobi": 10,
    "old_landline": 10,
    "new_landline": 11
}
