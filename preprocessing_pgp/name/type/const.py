"""
Constants for processing name type extraction
"""

import pandas as pd

from preprocessing_pgp.const import HDFS_BASE_PTH, hdfs

# ? IMPORTANT PATHS

_NAME_TYPE_LV1_PATH = f"{HDFS_BASE_PTH}/name_type/customer_type_extra.parquet"

_NAME_TYPE_LV2_PATH = f"{HDFS_BASE_PTH}/name_type/customer_type_lv2.parquet"

# ? DATA BY LEVELS
LV1_NAME_TYPE = pd.read_parquet(_NAME_TYPE_LV1_PATH, filesystem=hdfs)
LV2_NAME_TYPE = pd.read_parquet(_NAME_TYPE_LV2_PATH, filesystem=hdfs)
NAME_TYPE_DATA = {"lv1": LV1_NAME_TYPE, "lv2": LV2_NAME_TYPE}

# ? NAME TYPE REGEX
LV1_NAME_TYPE_WITHOUT_ACCENT_REGEX = (
    LV1_NAME_TYPE.query("~is_accented").groupby("ctype")["term"].apply(list).to_dict()
)
LV1_NAME_TYPE_ACCENTED_REGEX = (
    LV1_NAME_TYPE.query("is_accented").groupby("ctype")["term"].apply(list).to_dict()
)
LV2_NAME_TYPE_REGEX = LV2_NAME_TYPE.groupby("ctype")["term"].apply(list).to_dict()
NAME_TYPE_REGEX_DATA = {
    "lv1_accented": LV1_NAME_TYPE_ACCENTED_REGEX,
    "lv1_without_accent": LV1_NAME_TYPE_WITHOUT_ACCENT_REGEX,
    "lv2_accented": LV2_NAME_TYPE_REGEX,
    "lv2_without_accent": LV2_NAME_TYPE_REGEX,
}
