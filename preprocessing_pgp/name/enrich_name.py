import os
import argparse
import json
from time import time
from typing import List, Tuple

import pandas as pd
from tensorflow import keras
from tqdm import tqdm

from preprocessing_pgp.name.name_processing import NameProcessor
from preprocessing_pgp.name.model.transformers import TransformerModel
from preprocessing_pgp.preprocess import preprocess_df
from preprocessing_pgp.name.const import (
    NAME_SPLIT_PATH,
    MODEL_PATH,
    RULE_BASED_PATH
)

tqdm.pandas()


class EnrichName:
    """
    Wrap-up module to enrich and filling accent to names
    """

    def __init__(
        self,
        model_weight_path: str,
        vectorization_paths: Tuple[str, str],
        model_config_path: str,
        split_data_path: str,
        name_rb_pth: str
    ) -> None:
        start_time = time()
        self.model = self.load_model(
            model_weight_path,
            vectorization_paths,
            model_config_path
        )
        self.fname_rb = f'{name_rb_pth}/firstname_dict.parquet'
        self.mname_rb = f'{name_rb_pth}/middlename_dict.parquet'
        self.lname_rb = f'{name_rb_pth}/lastname_dict.parquet'
        self.name_processor = NameProcessor(
            self.model,
            self.fname_rb,
            self.mname_rb,
            self.lname_rb,
            split_data_path
        )
        self.total_load_time = time() - start_time

    def load_model(
        self,
        model_weight_path: str,
        vectorization_paths: Tuple[str, str],
        model_config_path: str
    ) -> TransformerModel:
        start_time = time()
        # ? Load config dict
        with open(model_config_path) as json_file:
            config_dict = json.load(json_file)

        # ? BUILD & LOAD MODEL
        source_vec_pth, target_vec_pth = vectorization_paths
        transformer = TransformerModel(
            source_vectorization=source_vec_pth,
            target_vectorization=target_vec_pth,
            config_dict=config_dict
        )
        transformer.build_model(
            optimizer=keras.optimizers.Adam(
                learning_rate=config_dict['LEARNING_RATE']),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        transformer.load_model_weights(model_weight_path)

        self.model_load_time = time() - start_time

        return transformer

    def refill_accent(
        self,
        name_df: pd.DataFrame,
        name_col: str
    ) -> pd.DataFrame:
        return self.name_processor.fill_accent(name_df, name_col)

    def enrich(
        self,
        raw_df: pd.DataFrame,
        name_col: str,
    ) -> pd.DataFrame:

        # ? Preprocess & Remove all non_human names
        # *-- Don't use Multi-processing cause without progress bar
        start_time = time()
        clean_names = preprocess_df(
            raw_df,
            name_col=name_col
        )
        self.preprocess_time = time() - start_time

        # ? Performing enrich names & filling accent
        # * Out df with columns: ['id_group', 'raw_name', 'enrich_name', 'final_name', 'similarity_score', 'is_human']
        start_time = time()
        # Apply filling accent
        predict_df = self.refill_accent(clean_names, name_col)

        self.enrich_time = time() - start_time

        return predict_df

    def get_time_report(self) -> pd.DataFrame:
        return pd.DataFrame({
            'model load time': [self.model_load_time],
            'total load time': [self.total_load_time],
            'preprocess time': [self.preprocess_time],
            'enrich time': [self.enrich_time]
        })


def process_enrich(
    raw_df: pd.DataFrame,
    name_col: str,
) -> Tuple[pd.DataFrame, EnrichName]:
    model_weight_path = f'{MODEL_PATH}/best_transformer_model.h5'
    vectorization_paths = (
        f'{MODEL_PATH}/vecs/source_vectorization_layer.pkl',
        f'{MODEL_PATH}/vecs/target_vectorization_layer.pkl'
    )
    model_config_path = f'{MODEL_PATH}/hp.json'

    enricher = EnrichName(
        model_weight_path=model_weight_path,
        vectorization_paths=vectorization_paths,
        model_config_path=model_config_path,
        split_data_path=NAME_SPLIT_PATH,
        name_rb_pth=RULE_BASED_PATH
    )

    final_df = enricher.enrich(
        raw_df,
        name_col
    )

    return final_df, enricher
