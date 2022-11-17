import pandas as pd
# import stanza
# from stanza import DownloadMethod
from unidecode import unidecode
from tqdm import tqdm

from preprocessing_pgp.const import NON_HUMAN_REG_LIST, REPLACE_HUMAN_REG_DICT

tqdm.pandas()


def replace_non_human_reg(name: str) -> str:
    for word, to_word in REPLACE_HUMAN_REG_DICT.items():
        name = name.replace(word, to_word)
    return name.strip()


def remove_non_person_with_rule(
    df: pd.DataFrame,
    name_col: str
) -> pd.DataFrame:
    print('Creating new non-accent names...')
    clean_df = df.copy()
    clean_df['without_accent'] = clean_df[name_col].progress_apply(unidecode)

    # Process with rule to remove non_human_names
    print('Cleaning special cases...')
    non_clean_mask = clean_df['without_accent'].progress_apply(
        lambda name: any(substr in name for substr in NON_HUMAN_REG_LIST))
    clean_df = clean_df[~non_clean_mask]
    clean_df = clean_df.reset_index(drop=True)
    print('\n\n')
    print('-'*20)
    print(f'{non_clean_mask.sum()} non human names detected with rule!')
    print('-'*20)
    print('\n\n')

    # Process to replace other non_meaning parts
    print('Replacing non-meaningful names...')
    clean_df[f'replaced_{name_col}'] = clean_df[name_col].progress_apply(
        replace_non_human_reg)

    replaced_mask = clean_df[f'replaced_{name_col}'] != clean_df[name_col]

    print('\n\n')
    print('-'*20)
    print(f'{replaced_mask.sum()} names replaced to beautify with rule!')
    print('-'*20)
    print('\n\n')

    # Clean up after working
    clean_df = clean_df.drop(columns=['without_accent', name_col])
    clean_df = clean_df.rename(columns={f'replaced_{name_col}': name_col})

    return clean_df, df[non_clean_mask]

