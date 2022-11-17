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
from preprocessing_pgp.utils import remove_non_accent_names
from preprocessing_pgp.extract_human import replace_non_human_reg, remove_non_person_with_rule


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

    # Remove non-human names
    # if extract_human:
    #     print("Perform full human-names extraction...")
    #     human_names, non_human_names = remove_non_person(
    #         basic_clean_names,
    #         name_col=name_col,
    #         model=human_extractor,
    #         multiprocessing=multiprocessing,
    #         n_cpu=n_cpu
    #     )
    # else:
    print("Perform rule human-names extraction...")
    human_names, non_human_names = remove_non_person_with_rule(
        basic_clean_names,
        name_col=name_col
    )

    return human_names, non_human_names


def preprocess_names(
    file_path,
    name_col='Name',
    save_file=False,
    save_path='data',
    save_name='data',
    save_fig=False,
    fig_save_path='plots',
    #  multiprocessing: bool = False,
    #  n_cpu: int = None
):
    """
    Clean & Preprocess DataFrame with 'Name' columns using method of standardize charset, accent typing (save plot of EDA if configured)

    Parameters
    ----------
    file_path : str
        Path to the data file of .parquet ext
    name_col : str
        Column name containing the names, by default 'Name'
    save_file : bool, optional
        Whether to save the datafile after clean and preprocess, by default False
    save_path : str, optional
        The path to the folder containing the preprocessed data file, by default 'data'
    save_name : str, optional
        The name of the saved data file, by default 'data.parquet'
    save_fig : bool, optional
        Whether to save EDA figures, by default False
    fig_save_path : str, optional
        The path to the folder containing those figures, by default 'plots'
    multiprocessing : bool, optional
        Whether to use multiprocessing cpu, by default False
    n_cpu : int, optional
        The number of cpu used when multiprocessing, by default None

    Returns
    -------
    pd.DataFrame
        The data after preprocessing and cleaning
    """

    # Read the data
    names = pd.read_parquet(file_path)
    print(f'Read data from {file_path} with shape {names.shape}\n')

    # Clean spare spaces
    print('Basic preprocessing names...')
    clean_names = names.copy()
    clean_names[name_col] = clean_names[name_col].progress_apply(
        basic_preprocess_name)

    # Count number of duplicate names & drop
    duplicated_mask = clean_names.duplicated(subset=[name_col])
    n_duplicate = clean_names[duplicated_mask].shape[0]
    print(f'Data contains {n_duplicate} duplicated values\n')

    # Counting the number of repeated names
    print('Top 5 repeated names')
    print(clean_names[duplicated_mask].value_counts().head(5), end='\n\n')

    # Dropping duplicated names
    print('Dropping duplicated values...')
    clean_names = clean_names.drop_duplicates(subset=[name_col])
    print(
        f'Shape of data after dropping duplicated values: {clean_names.shape}\n')

    # Remove accent typing to create data to train model
    print('Creating data for model...')

    # Clean the non-accent names and create training data
    # Don't remove 1-word name without accent
    clean_names_df, without_accent_names_df = remove_non_accent_names(
        clean_names,
        name_col=name_col,
        remove_single_name=False
    )

    # ? Saving figure of the number of data with accent typing and the one without accent typing
    # pandas_diff_barplot(clean_names_df,
    #                     first_column='with_accent',
    #                     second_column='without_accent',
    #                     correct_col='Without Accent',
    #                     wrong_col='With Accent',
    #                     savefile=save_fig,
    #                     filepath=f'{fig_save_path}/EDA Sentences With Accent Typing.png')

    # ? Show the number of data without accent typing in the original data
    n_without_accent = clean_names_df.shape[0]
    print(f'Number of sentences without accent typing: {n_without_accent}')

    # ? Re-check human names
    # human_extractor = HumanNameExtractor(language='vi')
    # human_names_df, non_human_names_df = remove_non_person(
    #     clean_names_df,
    #     'with_accent',
    #     human_extractor,
    #     multiprocessing=multiprocessing,
    #     n_cpu=n_cpu)
    human_names_df, non_human_names_df = preprocess_df(
        clean_names_df,
        name_col=name_col
    )

    # ? Saving the data as parquet extension
    if save_file:
        clean_names_df.to_parquet(f'{save_path}/{save_name}.parquet')
        without_accent_names_df.to_parquet(
            f'{save_path}/{save_name}_non_accent.parquet')
        human_names_df.to_parquet(f'{save_path}/{save_name}_human.parquet')
        non_human_names_df.to_parquet(
            f'{save_path}/{save_name}_non_human.parquet')

    return clean_names_df, without_accent_names_df, human_names_df, non_human_names_df


# if __name__ == '__main__':

#     # Initialize
#     parser = argparse.ArgumentParser(
#         description='Preprocessor for preprocess the data')

#     # Adding required arguments
#     parser.add_argument('-fp', '--file_path', type=str, required=True,
#                         help='Required path to data file in .parquet format')
#     parser.add_argument('-col', '--name_col', type=str, required=True,
#                         help='Required column name that contains the names')
#     parser.add_argument('--save_file', action='store_true',
#                         help='Whether to create vectorization for train data',
#                         default=False)
#     parser.add_argument('-sp', '--save_path', type=str, required=False,
#                         help='File save path if save option is enabled')
#     parser.add_argument('-sn', '--save_name', type=str, required=False,
#                         help='The name of the file to save')
#     parser.add_argument('--save_fig', action='store_true',
#                         help='Whether to save EDA figures',
#                         default=False)
#     parser.add_argument('-fsp', '--fig_save_path', type=str, required=False,
#                         help='Fig file save path if save fig option is enabled')
#     parser.add_argument('-mp', '--multiprocessing', action='store_true',
#                         help='Whether to save EDA figures',
#                         default=False)
#     parser.add_argument('--n_cpu',
#                         type=int, help='The number of cpu use for multiprocessing')

#     # Parse
#     args = parser.parse_args()

#     # Access
#     file_path = args.file_path
#     name_col = args.name_col
#     save_file = args.save_file
#     save_path = args.save_path
#     save_name = args.save_name
#     save_fig = args.save_fig
#     fig_save_path = args.fig_save_path
#     multiprocessing = args.multiprocessing
#     n_cpu = args.n_cpu

#     # Passing to trainer
#     dfs = preprocess_names(file_path,
#                            name_col=name_col,
#                            save_file=save_file,
#                            save_path=save_path,
#                            save_name=save_name,
#                            save_fig=save_fig,
#                            fig_save_path=fig_save_path,
#                            multiprocessing=multiprocessing,
#                            n_cpu=n_cpu)
