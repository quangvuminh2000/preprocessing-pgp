import multiprocessing as mp
from functools import partial
from typing import (
    Callable,
    List,
    Union,
    Tuple
)

import pandas as pd
import numpy as np
from unidecode import unidecode
from tqdm import tqdm

from preprocessing_pgp.const import (
    N_PROCESSES
)


tqdm.pandas()


def sentence_length(sentence: str) -> int:
    """
    Return the number of words in the sentence
    """
    return len(sentence.split())


def sep_display(sep: str = "\n") -> None:
    """
    Separator for output std
    """
    print(sep)


def is_empty_dataframe(data: pd.DataFrame) -> bool:
    """
    Check whether the dataframe is empty or not
    """
    return data.shape[0] == 0


def extract_null_values(
    data: pd.DataFrame,
    by_col: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extracting NULL values from specific DataFrame

    Parameters
    ----------
    data : pd.DataFrame
        Basic DataFrame
    by_col : str
        Column to separate Null values

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        Tuple of Non-Null DataFrame and Null DataFrame
    """

    null_data = data[data[by_col].isna()].copy()

    non_null_data = data.dropna(subset=[by_col]).copy()

    return non_null_data, null_data


def apply_multi_process(
    func: Callable,
    series: Union[pd.Series, str, np.ndarray],
    n_cores: int = N_PROCESSES
) -> List:
    """
    Process multi-processing on every items of series with provided func

    Parameters
    ----------
    func : Callable
        Function to traverse through series,
        must have 1 input and 1 output
    series : Optional[pd.Series]
        Any series | np.Array() | list
    n_cores : int
        The number of cores used to run parallel, by default half the cores will be used

    Returns
    -------
    List
        List of elements returned after apply the function
    """

    with mp.Pool(n_cores) as pool:
        output = list(pool.imap(func, series))

    return output


def parallelize_dataframe(
    data: pd.DataFrame,
    func: Callable,
    n_cores: int = N_PROCESSES,
    **kwargs
) -> pd.DataFrame:
    """
    Multi-processing on dataframe with provided function and additional function arguments

    Parameters
    ----------
    data : pd.DataFrame
        Any dataframe
    func : Callable
        Function to traverse through separate part of dataframe,
        input must contains the `dataframe` as the required argument
    n_cores : int
        The number of cores used to run parallel, by default half the cores will be used
    **kwargs
        Additional arguments for the function

    Returns
    -------
    pd.DataFrame
        Fully processed dataframe
    """

    if n_cores == 1:
        final_data = partial(func, **kwargs)(data)
    else:
        sub_data = np.array_split(data.copy(), n_cores)

        with mp.Pool(n_cores) as pool:
            final_data = pd.concat(pool.map(partial(func, **kwargs), sub_data))

    return final_data


def apply_progress_bar(
    func: Callable,
    series: pd.Series
) -> pd.Series:
    """
    Process apply with progress bar on every items of series with provided func

    Parameters
    ----------
    func : Callable
        Function to traverse through series, must have 1 input and 1 output
    series : pd.Series
        Any series of type pandas Series

    Returns
    -------
    pd.Series
        Series of elements returned after apply the function
    """

    return series.progress_apply(func)


def remove_non_accent_names(
    names_df: pd.DataFrame,
    name_col='name',
    remove_single_name=True
) -> pd.DataFrame:
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
