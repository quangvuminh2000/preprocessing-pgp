"""
Constants for processing name type extraction
"""

import os

import pandas as pd

# ? IMPORTANT PATHS
__NAME_TYPE_PATH = os.path.join(
    os.path.dirname(__file__),
    '../../',
    'data',
    'name_type'
)

_NAME_TYPE_LV1_PATH = os.path.join(
    __NAME_TYPE_PATH,
    'customer_type_lv1.parquet'
)

_NAME_TYPE_LV2_PATH = os.path.join(
    __NAME_TYPE_PATH,
    'customer_type_lv2.parquet'
)

# ? DATA BY LEVELS
LV1_NAME_TYPE = pd.read_parquet(_NAME_TYPE_LV1_PATH)
LV2_NAME_TYPE = pd.read_parquet(_NAME_TYPE_LV2_PATH)

# ? NAME TYPE DATA
NAME_TYPE_DATA = {
    'lv1': LV1_NAME_TYPE,
    'lv2': LV2_NAME_TYPE
}
