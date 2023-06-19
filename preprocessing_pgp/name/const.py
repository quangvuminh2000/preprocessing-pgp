"""
Module for essential constants in enrich name
"""
import os

import pandas as pd
from preprocessing_pgp.const import HDFS_BASE_PTH, hdfs

# ? MODEL PATHS
NAME_SPLIT_PATH = f"{HDFS_BASE_PTH}/name_split"

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    os.pardir,
    "data",
    "transformer_model",
    "trial-31",
)

# ? RULE-BASED PATH
# RULE_BASED_PATH = os.path.join(
#     os.path.dirname(__file__), os.pardir, "data", "rule_base", "students"
# )
RULE_BASED_PATH = f"{HDFS_BASE_PTH}/rule_base/students"

NICKNAME_PATH = f"{HDFS_BASE_PTH}/name_split/nicknames_boundary.parquet"

GENDER_MODEL_PATH = os.path.join(os.path.dirname(__file__), "../data/gender_model")

GENDER_MODEL_VERSION = "1.0"

PRONOUN_GENDER_RB_PATH = (
    f"{HDFS_BASE_PTH}/gender_model/rule_base/pronoun_gender_dict.parquet"
)

PRONOUN_GENDER_DF = pd.read_parquet(PRONOUN_GENDER_RB_PATH, filesystem=hdfs)
PRONOUN_GENDER_MAP = dict(
    zip(PRONOUN_GENDER_DF["pronoun"], PRONOUN_GENDER_DF["gender"])
)

NAME_ELEMENT_PATH = f"{HDFS_BASE_PTH}/name_split/name_elements.parquet"

MF_NAME_GENDER_RULE = pd.read_parquet(
    f"{HDFS_BASE_PTH}/rule_base/gender/mfname.parquet", filesystem=hdfs
)

BEFORE_FNAME_GENDER_RULE = pd.read_parquet(
    f"{HDFS_BASE_PTH}/rule_base/gender/before_fname.parquet", filesystem=hdfs
)

PRONOUN_REGEX = r"^(?:\bkh\b|\bkhach hang\b|\bchị\b|\bchi\b|\banh\b|\ba\b|\bchij\b|\bc\b|\be\b|\bem\b|\bcô\b|\bco\b|\bchú\b|\bbác\b|\bbac\b|\bme\b|\bdì\b|\bông\b|\bong\b|\bbà\b)\s+"
PRONOUN_REGEX_W_DOT = r"^(?:\bkh\b|\bkhach hang\b|\bchị\b|\bchi\b|\banh\b|\ba\b|\bchij\b|\bc\b|\be\b|\bem\b|\bcô\b|\bco\b|\bchú\b|\bbác\b|\bbac\b|\bme\b|\bdì\b|\bông\b|\bong\b|\bbà\b|\ba|\bc)[.,]"

REPLACE_HUMAN_REG_DICT = {"K HI": "", "Bs": "", "Ng.": "Nguyễn"}

BRIEF_NAME_DICT = {"nguyen": ["ng.", "n."], "do": ["d."], "pham": ["p."]}

# * NICKNAMES
NICKNAMES = pd.read_parquet(NICKNAME_PATH, filesystem=hdfs)
NICKNAME_REGEX = "|".join(
    [
        *NICKNAMES["name"].to_list(),
        *NICKNAMES[NICKNAMES["de_name"].str.split().str.len() > 1]["de_name"].to_list(),
    ]
)

# * NAME POSSIBLE ELEMENTS
NAME_ELEMENTS = pd.read_parquet(NAME_ELEMENT_PATH, filesystem=hdfs)
WITHOUT_ACCENT_ELEMENTS = set(NAME_ELEMENTS["without_accent"].unique())
WITH_ACCENT_ELEMENTS = set(NAME_ELEMENTS["with_accent"].unique())
