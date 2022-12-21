"""
File containing constants that are necessary for processing address extraction
"""

import os

import pandas as pd


# ? REGEX FOR ADDRESS
DICT_NORM_ABBREV_REGEX_KW = {
    'tp ': ['tp.', 'tp:'],
    'tt ': ['tt.', 'tt:'],
    'q ': ['q.', 'q:'],
    'h ': ['h.', 'h:'],
    'x ': ['x.', 'x:'],
    'p ': ['p.', 'p:']
}

DICT_NORM_CITY_DASH_REGEX = {
    'ba ria - vung tau': ['ba ria vung tau'],
    'br-vt': ['brvt'],
    'phan rang - thap cham': ['phan rang thap cham'],
    'pr-tc': ['prtc'],
    'thua thien hue': ['thua thien - hue']
}

ADDRESS_PUNCTUATIONS = ['-', '/']

# ? LEVEL METHODS
LV1_METHODS = ['lv1_norm', 'lv1_abbrev', 'lv1_prefix_im', 'lv1_nprefix_im']
LV2_METHODS = ['lv2_norm', 'lv2_abbrev', 'lv2_prefix_im', 'lv2_nprefix_im']
LV3_METHODS = ['lv3_norm', 'lv3_abbrev', 'lv3_prefix_im', 'lv3_nprefix_im']
METHOD_REFER_DICT = {
    1: LV1_METHODS,
    2: LV2_METHODS,
    3: LV3_METHODS
}

# ? LOCATION ENRICH DICTIONARY
__LOCATION_ENRICH_PATH = os.path.join(
    os.path.dirname(__file__),
    os.pardir,
    "data",
    "location_dict_enrich_address.parquet"
)
LOCATION_ENRICH_DICT = pd.read_parquet(__LOCATION_ENRICH_PATH)

# ? LOCATION CODE DICTIONARY
__LOCATION_CODE_PATH = os.path.join(
    os.path.dirname(__file__),
    os.pardir,
    "data",
    "location_dict_code.parquet"
)
LOCATION_CODE_DICT = pd.read_parquet(__LOCATION_CODE_PATH)

LEVEL_VI_COLUMN_DICT = {
    1: 'city_vi',
    2: 'district_vi',
    3: 'ward_vi'
}
LEVEL_CODE_COLUMN_DICT = {
    1: 'city_id',
    2: 'district_id',
    3: 'ward_id'
}

AVAIL_LEVELS = LEVEL_VI_COLUMN_DICT.keys()
