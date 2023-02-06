import os
import json
import argparse
from time import time
from string import capwords
from typing import List
import multiprocessing as mp

import pandas as pd
from tqdm import tqdm
from unidecode import unidecode
from tensorflow import keras

from preprocessing_pgp.name.split_name import NameProcess
from preprocessing_pgp.name.model.transformers import TransformerModel
from preprocessing_pgp.name.rulebase_name import rule_base_name
from preprocessing_pgp.name.utils import remove_nicknames

tqdm.pandas()


class NameProcessor:
    def __init__(self,
                 model: TransformerModel,
                 firstname_rb: pd.DataFrame,
                 middlename_rb: pd.DataFrame,
                 lastname_rb: pd.DataFrame,
                 base_path: str
                 ):
        self.model = model
        self.name_dicts = (firstname_rb, middlename_rb, lastname_rb)
        self.name_process = NameProcess(base_path)

    def predict_non_accent(self, name: str):
        if name is None:
            return None
        de_name = unidecode(name)

        # Keep case already have accent
        if name != de_name:
            return name

        # Only apply to case not having accent
        return capwords(self.model.predict(name))

    def fill_accent(self,
                    name_df: pd.DataFrame,
                    name_col: str,
                    #                     nprocess: int, # number of processes to multi-inference
                    ):
        predicted_name = name_df.copy(deep=True)
        orig_cols = predicted_name.columns

        # n_names = predicted_name.shape[0]
        # * Clean name before processing
        predicted_name[f'clean_{name_col}'] =\
            predicted_name[name_col].apply(self.name_process.CleanName)

        predicted_name = remove_nicknames(
            predicted_name,
            name_col=f'clean_{name_col}'
        )

        # print("Filling diacritics to names...")
        # start_time = time()
        predicted_name['predict'] = predicted_name[f'clean_{name_col}'].apply(
            self.predict_non_accent)
        # mean_predict_time = (time() - start_time) / n_names

        # print(f"\nAVG prediction time : {mean_predict_time}s")

        # print('\n\n')

        # print("Applying rule-based postprocess...")
        # start_time = time()
        predicted_name['final'] = predicted_name.apply(
            lambda row: rule_base_name(
                row['predict'], row[f'clean_{name_col}'], self.name_dicts),
            axis=1
        )

        # mean_rb_time = (time() - start_time) / n_names

        # print(f"\nAVG rb time : {mean_rb_time}s")

        # print('\n\n')

        return predicted_name[[*orig_cols, 'final', 'predict']]

    def unify_name(self,
                   name_df: pd.DataFrame,
                   name_col: str,
                   key_col: str,
                   keep_cols: List[str]):
        name_df[key_col] = name_df[key_col].astype('str')
        best_name_df = self.name_process.CoreBestName(
            name_df,
            name_col=name_col,
            key_col=key_col
        )

        return best_name_df[[key_col, name_col, 'best_name', 'similarity_score', *keep_cols]]
