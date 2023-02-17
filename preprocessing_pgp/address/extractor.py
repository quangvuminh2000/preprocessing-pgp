"""
File containing logic to extract a full address into 3 levels:

* Level 1: City, Countryside
* Level 2: District
* Level 3: Ward
"""
from time import time

import pandas as pd

from preprocessing_pgp.address.loc_process import generate_loc_code
from preprocessing_pgp.address.level_extractor import extract_vi_address_by_level
from preprocessing_pgp.address.preprocess import clean_vi_address
from preprocessing_pgp.utils import (
    sep_display,
    parallelize_dataframe,
    extract_null_values
)
from preprocessing_pgp.address.const import AVAIL_LEVELS


def extract_vi_address(
    data: pd.DataFrame,
    address_col: str,
    n_cores: int = 1
) -> pd.DataFrame:
    """
    Extract Vietnamese address by pattern to find 3 levels of address

    Parameters
    ----------
    data : pd.DataFrame
        The input raw data with address column
    address_col : str
        The name of the column containing addresses
    n_cores : int, optional
        The number of cores used to run parallel, by default 1 core will be used

    Returns
    -------
    pd.DataFrame
        The data with additional columns:

        * `cleaned_<address_col>` contains the unified and clean Vietnamese address
        * `level_1`: city, countryside found
        * `best_level_1`: beautified city, countryside found
        * `level_1_code`: code for city
        * `level_2`: district found
        * `best_level_2`: beautified district found
        * `level_2_code`: code for district
        * `level_3`: ward found
        * `best_level_3`: beautified ward found
        * `level_3_code`: code for ward
        * `remained address`: the remaining in the address
    """

    # * Select only address column
    orig_cols = data.columns
    address_data = data[[address_col]]

    # * Removing na addresses
    clean_address_data, na_address_data =\
        extract_null_values(
            address_data,
            by_col=address_col
        )

    # * Cleanse the address
    start_time = time()
    cleaned_data = clean_vi_address(clean_address_data, address_col)
    clean_time = time() - start_time
    print(f"Cleansing takes {int(clean_time)//60}m{int(clean_time)%60}s")
    sep_display()

    # * Feed the cleansed address to extract the level
    start_time = time()
    if n_cores == 1:  # Not using multi-processing
        extracted_data = extract_vi_address_by_level(
            cleaned_data,
            address_col=f'cleaned_{address_col}'
        )
    else:
        extracted_data = parallelize_dataframe(
            cleaned_data,
            extract_vi_address_by_level,
            n_cores=n_cores,
            address_col=f'cleaned_{address_col}'
        )
    extract_time = time() - start_time
    print(f"Extracting takes {int(extract_time)//60}m{int(extract_time)%60}s")
    sep_display()

    # * Generate location code for best level found
    start_time = time()
    best_lvl_cols = [f'best level {i}' for i in AVAIL_LEVELS]
    if n_cores == 1:
        generated_data = generate_loc_code(
            extracted_data,
            best_lvl_cols=best_lvl_cols
        )
    else:
        generated_data = parallelize_dataframe(
            extracted_data,
            generate_loc_code,
            n_cores=n_cores,
            best_lvl_cols=best_lvl_cols
        )
    code_gen_time = time() - start_time
    print(
        f"Code generation takes {int(code_gen_time)//60}m{int(code_gen_time)%60}s")

    # * Concat to original data
    final_data = pd.concat([generated_data, na_address_data])

    # * Concat with origin columns
    new_cols = [
        'level_1',
        'best_level_1',
        'level_1_code',
        'level_2',
        'best_level_2',
        'level_2_code',
        'level_3',
        'best_level_3',
        'level_3_code',
        'remained_address'
    ]
    final_data = pd.concat([
        data[orig_cols],
        final_data[new_cols]
    ], axis=1)

    return final_data
