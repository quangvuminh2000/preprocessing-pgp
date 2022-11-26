import os

import pandas as pd
import multiprocessing as mp
from tqdm import tqdm

from preprocessing_pgp.card.utils import (
    check_non_digit,
    check_card_length,
    is_valid_card,
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

    # * Check clean card
    card_unclean_mask = check_non_digit(card_df, card_col)

    card_df.loc[card_unclean_mask, "is_valid"] = False

    print("\n")
    print(f"# NON CLEAN CARD ID: {card_unclean_mask.sum()}")
    print("\n\n")

    # * Check card length

    (correct_length_df, possible_length_df, invalid_length_df) = check_card_length(
        card_df, card_col, valid_col="is_valid"
    )

    # * Check valid card
    with mp.Pool(PROCESSES) as pool:
        correct_length_df["is_valid"] = pool.map(
            is_valid_card, correct_length_df[card_col]
        )

    print("# CORRECT LENGTH CARD STATISTIC:")
    print(correct_length_df["is_valid"].value_counts())
    print("\n\n")

    with mp.Pool(PROCESSES) as pool:
        possible_length_df["is_valid"] = pool.map(
            is_valid_card, possible_length_df[f"clean_{card_col}"]
        )

    print("# POSSIBLE LENGTH CARD STATISTIC:")
    print(possible_length_df["is_valid"].value_counts())
    print("\n")

    # * Transform the clean card_code
    correct_length_df.loc[
        correct_length_df["is_valid"], f"clean_{card_col}"
    ] = correct_length_df[card_col]

    possible_length_df.loc[~possible_length_df["is_valid"], f"clean_{card_col}"] = None

    invalid_length_df[f"clean_{card_col}"] = None

    # * Merge to create final card DF
    final_card_df = pd.concat(
        [correct_length_df, possible_length_df, invalid_length_df]
    )

    # * Fill nas
    fill_cols = ["is_valid"]

    final_card_df[fill_cols] = final_card_df[fill_cols].fillna(False)

    return final_card_df
