import pandas as pd
import numpy as np
from unidecode import unidecode
from tqdm import tqdm


tqdm.pandas()

def sentence_length(sentence: str):
    return len(sentence.split())

def remove_non_accent_names(names_df: pd.DataFrame, name_col='name', remove_single_name=True) -> pd.DataFrame:
    """
    Remove non accent names inside the DF

    Parameters
    ----------
    names_df : pd.DataFrame
        The original names DF
    name_col : str, optional
        The column containing the data of names, by default 'name'
    remove_single_name : bool, optional
        Whether to remove a single word name, by default True

    Returns
    -------
    pd.DataFrame
        The clean final DF without any non_accent name
    """
    print("Decoding names...")
    names = names_df[name_col].copy()
    de_names = names.progress_apply(unidecode)

    with_accent_mask = names != de_names

    clean_names = names[with_accent_mask]
    clean_de_names = de_names[with_accent_mask]

    if not remove_single_name:
        len_name = names.apply(lambda name: len(name.split()))
        one_word_mask = len_name == 1
        clean_names = names[with_accent_mask | one_word_mask]
        clean_de_names = de_names[with_accent_mask | one_word_mask]

    clean_names_df = pd.DataFrame({
        'without_accent': clean_de_names,
        'with_accent': clean_names
    })

    without_accent_names_df = names_df[~with_accent_mask].copy()

    return clean_names_df, without_accent_names_df
