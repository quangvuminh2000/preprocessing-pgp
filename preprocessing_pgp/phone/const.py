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

# ? MEANINGFUL PHONE
# TODO Change to syntax detection for meaningful phone
DICT_MEANINGFUL_PHONE = {
    'Lục Quý': ['000000', '111111', '222222', '333333', '444444',
                '555555', '666666', '777777', '888888', '999999'],
    'Ngũ Quý': ['00000', '11111', '22222', '33333', '44444',
                '55555', '66666', '77777', '88888', '99999'],
    'Tứ Quý': ['0000', '1111', '2222', '3333', '4444',
               '5555', '6666', '7777', '8888', '9999'],
    'Tam Hoa': ['000', '111', '222', '333', '444',
                '555', '666', '777', '888', '999'],
    'Số Tiến': ['0123', '1234', '2345', '3456', '4567', '5678', '6789'],
    'Số Lùi': ['3210', '4321', '5432', '6543', '7654', '8765', '9876'], # Low in money
    'Lộc Phát': ['6868', '6886', '8686', '8668',
                 '886', '866'],
    'Thần Tài': ['3939', '3993', '9339', '9393',
                 '7979', '7997', '9779', '9797',
                 '3979', '9397', '9379', '9739',
                 '7939', '9793'],
    'Ông Địa': ['3838', '3883', '8338', '8383',
                '7878', '7887', '8778', '8787',
                '3878', '8387', '8378', '8738',
                '7838', '8783']
}

DICT_PHONE_TAIL_TYPE = {}
for phone_type, tail_formats in DICT_MEANINGFUL_PHONE.items():
    for fm in tail_formats:
        DICT_PHONE_TAIL_TYPE[fm] = phone_type
