import string

import numpy as np
import pandas as pd
from tqdm import tqdm
from unidecode import unidecode

# try:
#     try:
#         from utils import extract_firstname
#     except ImportError:
#         from scripts.train_script.utils import extract_firstname
# except ImportError:
#     from train_script.utils import extract_firstname

tqdm.pandas()


def evaluate_true_prediction(pred_df, accent_col='with_accent', pred_col='predict') -> float:
    """
    Calculate the true prediction between accent_col and pred_col of pred_df DataFrame

    Parameters
    ----------
    pred_df : pd.DataFrame
        The predicted DataFrame
    accent_col : str, optional
        The column name containing the diacritics names, by default 'with_accent'
    pred_col : str, optional
        The column name containing the predicted names, by default 'predict'

    Returns
    -------
    float
        The overall score of perfect predictions
    """
    mask_true_pred = pred_df[accent_col] == pred_df[pred_col]
    n_true_pred = pred_df[mask_true_pred].shape[0]

    score = (n_true_pred / pred_df.shape[0]) * 100

    return score


# def perform_true_prediction(df, model):
#     """
#     Predict true prediction which the predicted name must be exactly equal to the diacritics names

#     Parameters
#     ----------
#     df : pd.DataFrame
#         The original DF
#     model : Model
#         The model to perform prediction

#     Returns
#     -------
#     pd.DataFrame, float
#         Return the predicted DF containing the prediction and the evaluation score
#     """
#     print(f"Evaluating model on {len(df)} examples...")
#     predictions = df['without_accent'].progress_apply(model.predict)

#     print(f"Post processing...")
#     predict_df = df.copy()
#     predict_df['predict'] = predictions.progress_apply(
#         string.capwords)  # Capitalize words

#     score = evaluate_true_prediction(
#         predict_df, accent_col='with_accent', pred_col='predict')

#     return predict_df, score


# def change_evaluation(df, base_col='without_accent', pred_col='predict'):
#     base = df[base_col]
#     pred = df[pred_col]

#     de_pred = pred.progress_apply(unidecode)
#     sim_mask = base == de_pred

#     n_same = base[sim_mask].shape[0]
#     n_total = base.shape[0]

#     percent_change = 1 - (n_same/n_total)
#     total_changes = n_total - n_same

#     return percent_change, total_changes


# def n_differences(name_pred: str, name_base: str):
#     de_pred = unidecode(name_pred.strip())
#     de_base = unidecode(name_base.strip())

#     word_pred = np.array(de_pred.split())
#     word_base = np.array(de_base.split())

#     len_pred = word_pred.size
#     len_base = word_base.size

#     if len_pred == len_base:
#         return len_pred - np.sum(word_pred == word_base)

#     # Case not same # words
#     len_max = max(len_pred, len_base)
#     word_min = [word_pred, word_base][np.argmin([len_pred, len_base])]
#     word_max = [word_pred, word_base][np.argmax([len_pred, len_base])]

#     return len_max - np.sum(np.in1d(word_min, word_max))


# def evaluate_n_words_change(pred_df):
#     pred_diff_df = pred_df.copy()

#     pred_diff_df['n_differences'] = pred_diff_df.progress_apply(
#         lambda row: n_differences(row['predict'], row['without_accent']), axis=1)

#     return pred_diff_df


# def evaluate_accent_filling(pred_diff_df):
#     pred_no_change_df = pred_diff_df[pred_diff_df['n_differences'] == 0].copy()

#     score = evaluate_true_prediction(
#         pred_no_change_df, 'with_accent', 'predict')

#     return score


# def evaluate_firstname_true_prediction(pred_df: pd.DataFrame, accent_col='with_accent', pred_col='predict', without_accent_col='without_accent'):
#     firstname_pred_df = pred_df.copy()
#     firstname_pred_df[accent_col] = extract_firstname(
#         firstname_pred_df[accent_col])
#     firstname_pred_df[pred_col] = extract_firstname(
#         firstname_pred_df[pred_col])
#     firstname_pred_df[without_accent_col] = extract_firstname(
#         firstname_pred_df[without_accent_col])

#     score = evaluate_true_prediction(
#         firstname_pred_df, accent_col=accent_col, pred_col=pred_col)

#     return firstname_pred_df, score
