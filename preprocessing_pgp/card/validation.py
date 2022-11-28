import os

import pandas as pd
import numpy as np
import multiprocessing as mp
from tqdm import tqdm

from preprocessing_pgp.card.utils import (
    check_non_digit,
    check_card_length,
    is_valid_card,
    check_contain_digit,
    remove_spaces
)

tqdm.pandas()
PROCESSES = os.cpu_count() // 2


def verify_card(card_df: pd.DataFrame, card_col: str) -> pd.DataFrame:
    """
    Verify whether the card ids are valid or not

    Parameters
    ----------
    card_df : pd.DataFrame
        The input DF containing card id
    card_col : str
        The column contain card id

    Returns
    -------
    pd.DataFrame
        The final DF contains the columns that verify whether the card id is valid or not
    """

    # * Unify card to lower
    card_df[card_col] = card_df[card_col].str.lower()
    # * Removing spare spaces
    card_df[card_col] = card_df[card_col].apply(remove_spaces)

    # * Check clean card
    with mp.Pool(PROCESSES) as pool:
        card_all_digit_mask = pool.map(check_contain_digit, card_df[card_col])

    # card_unclean_mask = check_non_digit(card_df, card_col)
    card_all_digit_mask = np.array(card_all_digit_mask, dtype=np.bool8)

    card_df.loc[~card_all_digit_mask, "is_valid"] = False

    print()
    print(f"{'#'*5} CLEANSING {'#'*5}")
    print("\n")
    print(
        f"# CARD ID CONTAINS NON-DIGIT CHARACTERS: {card_df['is_valid'].notna().sum()}")
    print("\n")
    print("SAMPLE OF CARDS WITH NON-DIGIT CHARACTERS:")
    print(card_df[card_df['is_valid'].notna()].head(10))
    print("\n\n")

    # * Check card length

    (correct_length_df,
     possible_length_df,
     invalid_length_df) = check_card_length(
        card_df,
        card_col,
        valid_col="is_valid"
    )

    # * Check valid card
    # ? LENGTH 9 OR 12
    with mp.Pool(PROCESSES) as pool:
        correct_length_df["is_valid"] = pool.map(
            is_valid_card, correct_length_df[card_col]
        )

    print(f"# CARD OF LENGTH 9 OR 12: {correct_length_df.shape[0]}")

    print("STATISTIC:")
    stat_correct_length = correct_length_df["is_valid"].value_counts()
    print(f"# VALID: {stat_correct_length[True]}")
    print(f"# INVALID: {stat_correct_length[False]}")
    print("\n\n")

    # ? LENGTH 8 OR 11
    with mp.Pool(PROCESSES) as pool:
        possible_length_df["is_valid"] = pool.map(
            is_valid_card, possible_length_df[f"clean_{card_col}"]
        )

    print(f"# CARD OF LENGTH 8 OR 11: {possible_length_df.shape[0]}")

    print("STATISTIC:")
    stat_possible_length = possible_length_df["is_valid"].value_counts()
    print(f"# VALID: {stat_possible_length[True]}")
    print(f"# INVALID: {stat_possible_length[False]}")
    print("\n\n")

    # ? INVALID
    print(f"# CARD WITH OTHER LENGTH: {invalid_length_df.shape[0]}")

    # * Transform the clean card_code
    correct_length_df.loc[
        correct_length_df["is_valid"],
        f"clean_{card_col}"
    ] = correct_length_df[card_col]

    possible_length_df.loc[~possible_length_df["is_valid"],
                           f"clean_{card_col}"] = None

    invalid_length_df[f"clean_{card_col}"] = None

    # * Merge to create final card DF
    correct_length_df['is_valid'] = correct_length_df['is_valid'].astype(bool)
    possible_length_df['is_valid'] = possible_length_df['is_valid'].astype(
        bool)
    invalid_length_df['is_valid'] = invalid_length_df['is_valid'].astype(bool)
    final_card_df = pd.concat(
        [correct_length_df, possible_length_df, invalid_length_df]
    )

    # * Fill nas
    fill_cols = ["is_valid"]

    final_card_df[fill_cols] = final_card_df[fill_cols].fillna(False)

    # * Check if passport is found
    final_card_df['is_passport'] = False
    passport_mask = (
        (final_card_df['is_valid'] == False) &
        (final_card_df[card_col].str.contains(r'^[a-z]\d{7}$'))
    )
    final_card_df.loc[
        passport_mask,
        ['is_passport', 'is_valid']
    ] = True
    final_card_df.loc[
        passport_mask,
        f'clean_{card_col}'
    ] = final_card_df.loc[
        passport_mask,
        card_col
    ].str.upper()

    print(f"# PASSPORT FOUND: {passport_mask.sum()}")
    print("\n")
    print("SAMPLE OF PASSPORT:")
    print(final_card_df[passport_mask].head(10))
    print("\n\n")

    general_valid_statistic = final_card_df["is_valid"].value_counts()
    print(f"{'#'*5} GENERAL CARD ID REPORT {'#'*5}")
    print()
    print(f"COHORT SIZE: {final_card_df.shape[0]}")
    print(f"VALID CARD: {general_valid_statistic[True]}")
    print(f"INVALID CARD: {general_valid_statistic[False]}")
    print(f"PASSPORT: {final_card_df.query('is_passport').shape[0]}")
    print()

    return final_card_df
