import sys

import pandas as pd
from tqdm import tqdm

from preprocessing_pgp.phone.extractor import process_convert_phone

sys.path.append('/bigdata/fdp/cdp/source/core_profile/preprocess/utils')
from const import (
    hdfs,
    CENTRALIZE_PATH,
    UTILS_PATH,
    PRODUCT_PATH,
)


# ? MAIN FUNCTION
def daily_enhance_card(
    day: str,
    n_cores: int = 1
):
    pass



