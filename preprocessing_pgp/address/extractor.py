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
from preprocessing_pgp.utils import sep_display, parallelize_dataframe
from preprocessing_pgp.address.const import AVAIL_LEVELS


def extract_vi_address(data: pd.DataFrame, address_col: str) -> pd.DataFrame:
    """
    Extract Vietnamese address by pattern to find 3 levels of address

    Parameters
    ----------
    data : pd.DataFrame
        The input raw data with address column
    address_col : str
        The name of the column containing addresses

    Returns
    -------
    pd.DataFrame
        The data with additional columns:

        * `cleaned_<address_col>` contains the unified and clean Vietnamese address
        * `level 1`: city, countryside found
        * `level 2`: district found
        * `level 3`: ward found
        * `remained address`: the remaining in the address
    """

    # * Cleanse the address
    start_time = time()
    cleaned_data = clean_vi_address(data, address_col)
    clean_time = time() - start_time
    print(f"Cleansing takes {int(clean_time)//60}m{int(clean_time)%60}s")
    sep_display()

    # * Feed the cleansed address to extract the level
    start_time = time()
    extracted_data = parallelize_dataframe(
        cleaned_data,
        extract_vi_address_by_level,
        address_col=f'cleaned_{address_col}'
    )
    extract_time = time() - start_time
    print(f"Extracting takes {int(extract_time)//60}m{int(extract_time)%60}s")
    sep_display()

    # * Generate location code for best level found
    start_time = time()
    generated_data = parallelize_dataframe(
        extracted_data,
        generate_loc_code,
        best_lvl_cols=[f'best level {i}' for i in AVAIL_LEVELS]
    )
    code_gen_time = time() - start_time
    print(
        f"Code genration takes {int(code_gen_time)//60}m{int(code_gen_time)%60}s")

    return generated_data
