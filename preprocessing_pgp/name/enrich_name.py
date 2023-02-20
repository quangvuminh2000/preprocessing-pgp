import json
from time import time
from typing import Tuple
import warnings
import logging

import pandas as pd
from tensorflow import keras
from tqdm import tqdm
from halo import Halo

from preprocessing_pgp.name.name_processing import NameProcessor
from preprocessing_pgp.name.model.transformers import TransformerModel
from preprocessing_pgp.name.type.extractor import process_extract_name_type
from preprocessing_pgp.name.const import (
    MODEL_PATH,
    RULE_BASED_PATH
)
from preprocessing_pgp.utils import (
    sep_display,
    parallelize_dataframe,
)

tqdm.pandas()
warnings.filterwarnings("ignore")
logger = logging.getLogger()
logger.setLevel(logging.CRITICAL)


class EnrichName:
    """
    Wrap-up module to enrich and filling accent to names
    """

    def __init__(
        self,
        model_weight_path: str,
        vectorization_paths: Tuple[str, str],
        model_config_path: str,
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
        )
        # Timing
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

    def get_time_report(self) -> pd.DataFrame:
        return pd.DataFrame({
            'model load time': [self.model_load_time],
            'total load time': [self.total_load_time],
        })


@Halo(
    text='Enriching Names',
    color='cyan',
    spinner='dots7',
    text_color='magenta'
)
def enrich_clean_data(
    clean_df: pd.DataFrame,
    name_col: str,
) -> pd.DataFrame:
    """
    Applying the model of filling accent to cleaned Vietnamese names

    Parameters
    ----------
    clean_df : pd.DataFrame
        The dataframe containing cleaned names
    name_col : str
        The column name that holds the raw names

    Returns
    -------
    pd.DataFrame
        The final dataframe that contains:

        * `name_col`: raw names -- input name column
        * `predict`: predicted names using model only
        * `final`: beautified version of prediction with additional rule-based approach
    """
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
        name_rb_pth=RULE_BASED_PATH
    )

    final_df = enricher.refill_accent(
        clean_df,
        name_col
    )

    return final_df


def process_enrich(
    data: pd.DataFrame,
    name_col: str = 'name',
    n_cores: int = 1
) -> pd.DataFrame:
    """
    Applying the model of filling accent to non-accent Vietnamese names
    received from the `raw_df` at `name_col` column.

    Parameters
    ----------
    data : pd.DataFrame
        The dataframe containing raw names
    name_col : str
        The column name that holds the raw names, by default 'name'
    n_cores : int
        The number of cores used to run parallel, by default 1 core is used

    Returns
    -------
    pd.DataFrame
        The final dataframe that contains:

        * `customer_type`: the type of customer extracted from name
        * `predict`: predicted names using model only
        * `final`: beautified version of prediction with additional rule-based approach
    """
    orig_cols = data.columns

    # * Na names & filter out name col
    na_data = data[data[name_col].isna()][[name_col]]
    cleaned_data = data[data[name_col].notna()][[name_col]]

    # * Extracting customer type -- Only enrich 'customer' type
    cleaned_data = process_extract_name_type(
        cleaned_data,
        name_col,
        n_cores=n_cores
    )
    customer_data = cleaned_data.query('customer_type == "customer"')
    #                    | (cleaned_data[name_col].str.split(' ').str.len() > 3))
    non_customer_data = cleaned_data.query('customer_type != "customer"')

    # # Clean names -- Not Needed
    # start_time = time()
    # if n_cores == 1:
    #     customer_data = preprocess_df(
    #         customer_data,
    #         name_col=name_col
    #     )
    # else:
    #     customer_data = parallelize_dataframe(
    #         customer_data,
    #         preprocess_df,
    #         n_cores=n_cores,
    #         name_col=name_col
    #     )
    # clean_time = time() - start_time
    # print(f"Cleansing takes {int(clean_time)//60}m{int(clean_time)%60}s")
    # sep_display()

    # Enrich names
    start_time = time()
    if n_cores == 1:
        enriched_data = enrich_clean_data(
            customer_data,
            name_col=name_col
        )
    else:
        enriched_data = parallelize_dataframe(
            customer_data,
            enrich_clean_data,
            n_cores=n_cores,
            name_col=name_col
        )
    enrich_time = time() - start_time
    print(f"Enrich names takes {int(enrich_time)//60}m{int(enrich_time)%60}s")
    sep_display()

    # * Concat na data
    final_data = pd.concat([
        enriched_data,
        non_customer_data,
        na_data
    ])

    # * Concat with original cols
    new_cols = [
        'customer_type',
        'predict',
        'final',
        'last_name',
        'middle_name',
        'first_name',
    ]
    final_data = pd.concat([data[orig_cols], final_data[new_cols]], axis=1)

    return final_data
