import string
import re
import argparse
import os
import sys
from string import punctuation
from typing import Tuple

import pandas as pd
from tqdm import tqdm

from preprocessing_pgp.accent_typing_formatter import reformat_vi_sentence_accent
from preprocessing_pgp.unicode_converter import minimal_convert_unicode
from preprocessing_pgp.extract_human import replace_non_human_reg


# Enable progress-bar with pandas operations
tqdm.pandas()

_dir = "/".join(os.path.split(os.getcwd()))
if _dir not in sys.path:
    sys.path.append(_dir)


def remove_special_chars(sentence: str) -> str:
    """
    Removing special characters from sentence

    Parameters
    ----------
    sentence : str
        The input sentence can contains many special characters and alpha characters

    Returns
    -------
    str
        The sentence contains only alpha characters
    """
    return sentence.translate(str.maketrans('', '', punctuation))


def remove_spare_spaces(sentence: str) -> str:
    """
    Removing spare spaces inside a sentence

    Parameters
    ----------
    sentence : str
        Sentence to clean spare spaces

    Returns
    -------
    str
        Cleaned sentence
    """

    # Remove spaces in between
    sentence = re.sub(' +', ' ', sentence)
    # Remove additional leading & trailing spaces
    sentence = sentence.strip()

    return sentence


def format_caps_word(sentence: str) -> str:
    """
    Format the sentence into capital format

    Parameters
    ----------
    sentence : str
        The input sentences

    Returns
    -------
    str
        Capitalized sentence of the input sentence
    """
    caps_sen = sentence.lower()
    caps_sen = string.capwords(caps_sen)

    return caps_sen


def basic_preprocess_name(name: str) -> str:
    """
    Preprocess names based on these steps:

        1. Remove spare spaces
        2. Format name into capitalized
        3. Change name to Unicode compressed format
        4. Change name to same old accent typing format
        5. Re-capitalized the word

    Parameters
    ----------
    name : str
        The input raw name

    Returns
    -------
    str
        The preprocessed name
    """
    # Remove Spare Spaces
    clean_name = remove_spare_spaces(name)

    # Remove Special Characters
    clean_name = remove_special_chars(clean_name)

    # Format Caps
    caps_name = format_caps_word(clean_name)

    # Change to same VN charset -> Unicode compressed
    unicode_clean_name = minimal_convert_unicode(caps_name)

    # Change to same accent typing -> old type
    old_unicode_clean_name = reformat_vi_sentence_accent(unicode_clean_name)

    old_unicode_clean_name = format_caps_word(old_unicode_clean_name)

    return old_unicode_clean_name


def clean_name_cdp(name: str) -> str:
    if name == None:
        return None
    clean_name = basic_preprocess_name(name)
    clean_name = replace_non_human_reg(name)
    return clean_name


def preprocess_df(
    df: pd.DataFrame,
    # human_extractor: HumanNameExtractor,
    name_col: str = 'Name',
    # extract_human: bool = False,
    # multiprocessing: bool = False,
    # n_cpu: int = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Basic Preprocessing data
    print("Basic pre-processing names...")
    basic_clean_names = df.copy()
    basic_clean_names[f'clean_{name_col}'] = basic_clean_names[name_col].progress_apply(
        basic_preprocess_name)

    clean_name_mask = basic_clean_names[f'clean_{name_col}'] != basic_clean_names[name_col]
    print('\n\n')
    print('-'*20)
    print(f'{clean_name_mask.sum()} names have been clean!')
    print('-'*20)
    print('\n\n')

    basic_clean_names = basic_clean_names.drop(columns=[name_col])
    basic_clean_names = basic_clean_names.rename(columns={
        f'clean_{name_col}': name_col
    })

    return basic_clean_names
