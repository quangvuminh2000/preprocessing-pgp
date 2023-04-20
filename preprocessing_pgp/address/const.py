"""
File containing constants that are necessary for processing address extraction
"""

import os

import pandas as pd

# ? REGEX FOR ADDRESS
DICT_NORM_ABBREV_REGEX_KW = {
    "Tp ": ["Tp.", "Tp:"],
    "Tt ": ["Tt.", "Tt:"],
    "Q ": ["Q.", "Q:"],
    "H ": ["H.", "H:"],
    "X ": ["X.", "X:"],
    "P ": ["P.", "P:"],
}

DICT_NORM_CITY_DASH_REGEX = {
    " Bà Rịa - Vũng Tàu ": [
        "Ba Ria Vung Tau",
        "Bà Rịa-Vũng Tàu",
        "Ba Ria-Vung Tau",
    ],
    " Br-Vt ": ["Brvt", "Br - Vt"],
    " Phan Rang-Tháp Chàm ": [
        "Phan Rang Thap Cham",
        "Phan Rang - Tháp Chàm",
        "Phan Rang - Thap Cham",
    ],
    " Pr-Tc ": ["Prtc", "Pr - Tc"],
    " Thua Thien Hue ": ["Thua Thien - Hue", "Hue"],
    " Hồ Chí Minh ": ["Sài Gòn", "Sai Gon", "Tphcm", "Hcm", "Sg"],
    " Đà Nẵng ": [
        "Quang Nam-Da Nang",
        "Quảng Nam-Đà Nẵng",
        "Quang Nam - Da Nang",
        "Quảng Nam - Đà Nẵng",
    ],
    " Thành Phố ": ["Tp"],
}

ADDRESS_PUNCTUATIONS = ["-", "/", ","]

# ? LEVEL METHODS
LV1_METHODS = [
    "lv1_title",
    "lv1_title_norm",
    "lv1_norm",
    "lv1_abbrev",
    "lv1_prefix_im",
    "lv1_nprefix_im",
]
LV2_METHODS = [
    "lv2_title",
    "lv2_norm",
    "lv2_prefix_im",
    "lv2_title_norm",
    "lv2_abbrev",
]
LV3_METHODS = [
    "lv3_title",
    "lv3_norm",
    "lv3_prefix_im",
    "lv3_title_norm",
    "lv3_abbrev",
]
METHOD_REFER_DICT = {1: LV1_METHODS, 2: LV2_METHODS, 3: LV3_METHODS}

# ? LOCATION ENRICH DICTIONARY
__LOCATION_ENRICH_PATH = os.path.join(
    os.path.dirname(__file__),
    os.pardir,
    "data",
    "location_dict_enrich_address.parquet",
)
LOCATION_ENRICH_DICT = pd.read_parquet(__LOCATION_ENRICH_PATH)

# ? LOCATION CODE DICTIONARY
__LOCATION_CODE_PATH = os.path.join(
    os.path.dirname(__file__), os.pardir, "data", "location_dict_code.parquet"
)
LOCATION_CODE_DICT = pd.read_parquet(__LOCATION_CODE_PATH)

LEVEL_VI_COLUMN_DICT = {1: "city_vi", 2: "district_vi", 3: "ward_vi"}
LEVEL_CODE_COLUMN_DICT = {1: "city_id", 2: "district_id", 3: "ward_id"}

AVAIL_LEVELS = LEVEL_VI_COLUMN_DICT.keys()
