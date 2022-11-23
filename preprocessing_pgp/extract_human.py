import pandas as pd
from unidecode import unidecode
from tqdm import tqdm

from preprocessing_pgp.const import REPLACE_HUMAN_REG_DICT

tqdm.pandas()


def replace_non_human_reg(name: str) -> str:
    for word, to_word in REPLACE_HUMAN_REG_DICT.items():
        name = name.replace(word, to_word)
    return name.strip()

