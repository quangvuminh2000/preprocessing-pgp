"""
Module to predict for gender from name
"""
from time import time

import pandas as pd

from preprocessing_pgp.name.type.extractor import process_extract_name_type
from preprocessing_pgp.name.preprocess import preprocess_df
from preprocessing_pgp.name.model.lstm import predict_gender_from_name
from preprocessing_pgp.utils import (
    parallelize_dataframe,
    sep_display
)


def process_predict_gender(
    data: pd.DataFrame,
    name_col: str = 'name',
    n_cores: int = 1,
    logging_info: bool = True
) -> pd.DataFrame:
    """
    Process predict gender from names

    Parameters
    ----------
    data : pd.DataFrame
        The data contains the column with username
    name_col : str, optional
        The column name contains records of username, by default 'name'
    n_cores : int, optional
        The number of core to process, by default 1
    logging_info : bool, optional
        Whether to log info while running, by default True

    Returns
    -------
    pd.DataFrame
        Data with additional column:
        * `gender_predict`: predicted gender from name
    """
    name_data = data[[name_col]].dropna()
    nan_data = data[data[name_col].isna()][[name_col]]
    orig_cols = data.columns

    # * Clean name data
    start_time = time()
    cleaned_name_data = parallelize_dataframe(
        name_data,
        preprocess_df,
        n_cores=n_cores,
        name_col=name_col
    )
    clean_time = time() - start_time
    if logging_info:
        print(
            f"Cleansing names takes {int(clean_time)//60}m{int(clean_time)%60}s")
        sep_display()

    # * Get customer type
    cleaned_name_data = process_extract_name_type(
        cleaned_name_data,
        name_col=name_col,
        n_cores=n_cores,
        logging_info=False
    )

    # * Only predict for customer's name
    customer_mask = cleaned_name_data['customer_type'] == 'customer'
    customer_name_data = cleaned_name_data[customer_mask]
    non_customer_name_data = cleaned_name_data[~customer_mask]

    # * Predict gender
    start_time = time()
    predicted_name_data = parallelize_dataframe(
        customer_name_data,
        predict_gender_from_name,
        n_cores=n_cores,
        name_col=name_col
    )
    predict_time = time() - start_time
    if logging_info:
        print(
            f"Predicting gender takes {int(predict_time)//60}m{int(predict_time)%60}s")
        sep_display()

    # * Concat to generate final data
    final_data = pd.concat(
        [predicted_name_data, nan_data, non_customer_name_data])
    final_data = pd.concat(
        [data[orig_cols], final_data[['gender_predict']]], axis=1)

    return final_data
