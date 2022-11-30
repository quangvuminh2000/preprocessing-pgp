import os

import pandas as pd
import numpy as np


# * IMPORTANT CONSTANTS
CURRENT_YEAR = 22
LATEST_YEAR_DRIVER_LICENSE = 45


# * Personal Identification Card
__OLD_CODE_PATH = os.path.join(
    os.path.dirname(__file__), os.pardir, "data", "old_codes.parquet"
)
__NEW_CODE_PATH = os.path.join(
    os.path.dirname(__file__), os.pardir, "data", "new_codes.parquet"
)

POSSIBLE_GENDER_NUM = ['0', '1', '2', '3']
OLD_CODE_NUMS = pd.read_parquet(__OLD_CODE_PATH)['code'].values
NEW_CODE_NUMS = pd.read_parquet(__NEW_CODE_PATH)['code'].values
OLD_CODE_LENGTH = 9
NEW_CODE_LENGTH = 12

# * Driver License Card
'''
Documentation: https://tuhocvachiase.com/y-nghia-day-12-so-tren-bang-lai-xe-the-pet-xx-y-zz-1234567/
'''
DRIVER_LICENSE_ID_REGION_CODES = np.array(
    [code[1:] for code in NEW_CODE_NUMS], dtype=object)

INVALID_DRIVER_LICENSE_PASSING_YEAR = np.arange(CURRENT_YEAR+1, LATEST_YEAR_DRIVER_LICENSE, dtype=object)

DRIVER_LICENSE_LENGTH = 12
