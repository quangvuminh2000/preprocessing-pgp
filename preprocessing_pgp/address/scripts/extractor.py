"""
File containing logic to extract a full address into 3 levels:

* Level 1: City, Countryside
* Level 2: District
* Level 3: Ward
"""
from time import time

import pandas as pd

from preprocessing_pgp.address.scripts.level_extractor import extract_vi_address_by_level
from preprocessing_pgp.address.scripts.preprocess import clean_vi_address
from preprocessing_pgp.utils import sep_display, parallelize_dataframe


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
    print("Cleansing address...")
    cleaned_data = clean_vi_address(data, address_col)
    sep_display()

    # * Feed the cleansed address to extract the level
    start_time = time()
    print("Extracting address...")
    extracted_data = parallelize_dataframe(
        cleaned_data,
        extract_vi_address_by_level,
        address_col=f'cleaned_{address_col}'
    )
    # extracted_data = extract_vi_address_by_level(cleaned_data, f'cleaned_{address_col}')
    extract_time = time() - start_time
    print(f"Extracting takes {int(extract_time)//60}m{int(extract_time)%60}s")

    return extracted_data
