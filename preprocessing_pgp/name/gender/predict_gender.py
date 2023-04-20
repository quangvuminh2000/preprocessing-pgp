"""
Module to predict for gender from name
"""
from time import time

import pandas as pd
from preprocessing_pgp.name.const import (
    BEFORE_FNAME_GENDER_RULE,
    MF_NAME_GENDER_RULE,
    PRONOUN_GENDER_MAP,
)
from preprocessing_pgp.name.model.lstm import predict_gender_from_name
from preprocessing_pgp.name.preprocess import get_name_pronoun
from preprocessing_pgp.name.type.extractor import process_extract_name_type
from preprocessing_pgp.utils import parallelize_dataframe


def process_predict_gender(
    data: pd.DataFrame,
    name_col: str = "name",
    pronoun_col: str = "pronoun",
    n_cores: int = 1,
    logging_info: bool = True,
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
    orig_cols = data.columns
    if pronoun_col not in orig_cols:
        name_data = data[data[name_col].notna()][[name_col]]
        nan_data = data[data[name_col].isna()][[name_col]]
    else:
        name_data = data[data[name_col].notna()][[name_col, pronoun_col]]
        nan_data = data[data[name_col].isna()][[name_col, pronoun_col]]

    # # * Clean name data
    # if logging_info:
    #     print(">>> Cleansing name: ", end='')
    # start_time = time()
    # cleaned_name_data = parallelize_dataframe(
    #     name_data,
    #     preprocess_df,
    #     n_cores=n_cores,
    #     name_col=name_col
    # )
    # clean_time = time() - start_time
    # if logging_info:
    #     print(f"{int(clean_time)//60}m{int(clean_time)%60}s")
    # ? Extracting pronoun
    if pronoun_col not in orig_cols:
        if logging_info:
            print("Not find pronoun -- Try extracting from name")
            print(">>> Extracting pronouns: ", end="")
        start_time = time()
        name_data[pronoun_col] = name_data[name_col].apply(get_name_pronoun)
        pronoun_time = time() - start_time
        if logging_info:
            print(f"{int(pronoun_time)//60}m{int(pronoun_time)%60}s")

    # * Get customer type
    cleaned_name_data = process_extract_name_type(
        name_data, name_col=name_col, n_cores=n_cores, logging_info=False
    )

    # * Only predict for customer's name
    customer_mask = cleaned_name_data["customer_type"] == "customer"
    customer_name_data = cleaned_name_data[customer_mask]
    non_customer_name_data = cleaned_name_data[~customer_mask]

    # * Predict gender
    if logging_info:
        print(">>> Predicting gender: ", end="")
    start_time = time()
    predicted_name_data = parallelize_dataframe(
        customer_name_data,
        predict_gender_from_name,
        n_cores=n_cores,
        name_col=name_col,
    )
    predict_time = time() - start_time
    if logging_info:
        print(f"{int(predict_time)//60}m{int(predict_time)%60}s")

    # * Fill pronoun gender
    if logging_info:
        print("\t>> Fill pronoun gender")
    predicted_name_data["pronoun_gender"] = predicted_name_data[
        pronoun_col
    ].map(PRONOUN_GENDER_MAP)

    predicted_name_data["mf_name_gender"] = None
    # * MFNAMES Rule
    mfname_regex_female = (
        MF_NAME_GENDER_RULE[MF_NAME_GENDER_RULE["gender"] == "F"]["mfname"]
        + "$"
    )
    mfname_regex_female = "|".join(mfname_regex_female.unique())

    mfname_regex_male = (
        MF_NAME_GENDER_RULE[MF_NAME_GENDER_RULE["gender"] == "M"]["mfname"]
        + "$"
    )
    mfname_regex_male = "|".join(mfname_regex_male.unique())

    predicted_name_data.loc[
        predicted_name_data[name_col]
        .str.lower()
        .str.contains(f"(?i){mfname_regex_female}", regex=True, na=False),
        "mf_name_gender",
    ] = "F"
    predicted_name_data.loc[
        predicted_name_data[name_col]
        .str.lower()
        .str.contains(f"(?i){mfname_regex_male}", regex=True, na=False),
        "mf_name_gender",
    ] = "M"

    # * BEFORE Fnames rule
    predicted_name_data["before_fname_gender"] = None
    predicted_name_data["before_fname"] = (
        predicted_name_data[name_col].str.split().str.get(-2).str.lower()
    )
    before_fnames_regex = r"(?i) phương$| anh$| linh$| hà$| ngọc$| thư$| an$| châu$| khánh$| thương$| tú$| hạnh$| hiền$| thanh$| xuân$| minh$| quỳnh$| giang$| nguyên$"
    predicted_name_data.loc[
        predicted_name_data[name_col]
        .str.lower()
        .str.contains(f"(?i){before_fnames_regex}", regex=True, na=False)
        & predicted_name_data["before_fname"].isin(
            BEFORE_FNAME_GENDER_RULE.query('gender == "F"')[
                "before_fname"
            ].unique()
        ),
        "before_fname_gender",
    ] = "F"

    predicted_name_data.loc[
        predicted_name_data[name_col]
        .str.lower()
        .str.contains(f"(?i){before_fnames_regex}", regex=True, na=False)
        & predicted_name_data["before_fname"].isin(
            BEFORE_FNAME_GENDER_RULE.query('gender == "M"')[
                "before_fname"
            ].unique()
        ),
        "before_fname_gender",
    ] = "M"

    # * Prioritize pronoun gender
    if logging_info:
        print("\t>> Prioritize pronoun gender")
    predicted_name_data.loc[
        predicted_name_data["pronoun_gender"].notna(), "gender_predict"
    ] = predicted_name_data["pronoun_gender"]
    predicted_name_data.loc[
        predicted_name_data["pronoun_gender"].isna()
        & predicted_name_data["mf_name_gender"].notna(),
        "gender_predict",
    ] = predicted_name_data["mf_name_gender"]
    predicted_name_data.loc[
        predicted_name_data["pronoun_gender"].isna()
        & predicted_name_data["mf_name_gender"].isna()
        & predicted_name_data["before_fname_gender"].notna(),
        "gender_predict",
    ] = predicted_name_data["before_fname_gender"]

    # * Concat to generate final data
    new_cols = ["gender_predict", "gender_score"]
    nan_data[new_cols] = None
    final_data = pd.concat(
        [predicted_name_data, nan_data, non_customer_name_data]
    )
    final_data = pd.concat([data[orig_cols], final_data[new_cols]], axis=1)

    return final_data
