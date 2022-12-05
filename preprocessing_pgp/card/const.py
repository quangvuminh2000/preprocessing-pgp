import os

import pandas as pd
import numpy as np


# * IMPORTANT CONSTANTS
CURRENT_YEAR = 22
OLDEST_YEAR_DRIVER_LICENSE = 45
N_PROCESSES = os.cpu_count() // 2


# * PERSONAL IDENTIFICATION CARD
__OLD_PID_CODE_PATH = os.path.join(
    os.path.dirname(__file__), os.pardir, "data", "old_codes.parquet"
)
__NEW_PID_CODE_PATH = os.path.join(
    os.path.dirname(__file__), os.pardir, "data", "new_codes.parquet"
)

POSSIBLE_GENDER_NUM = ['0', '1', '2', '3']
OLD_PID_REGION_CODE_NUMS = pd.read_parquet(__OLD_PID_CODE_PATH)['code'].values
NEW_PID_REGION_CODE_NUMS = pd.read_parquet(__NEW_PID_CODE_PATH)['code'].values
OLD_PID_CODE_LENGTH = 9
NEW_PID_CODE_LENGTH = 12

# * PASSPORT CARD
PASSPORT_LENGTH = 8
PASSPORT_PATTERN = r'^[a-z]\d{7}$'

# * DRIVER LICENSE CARD
'''
Documentation: https://tuhocvachiase.com/y-nghia-day-12-so-tren-bang-lai-xe-the-pet-xx-y-zz-1234567/
'''
DRIVER_LICENSE_ID_REGION_CODES = np.array(
    [code[1:] for code in NEW_PID_REGION_CODE_NUMS], dtype=object)

INVALID_DRIVER_LICENSE_PASSING_YEAR =\
    np.array(
        [range(CURRENT_YEAR+1, OLDEST_YEAR_DRIVER_LICENSE)],
        dtype=object)

DRIVER_LICENSE_LENGTH = 12
INVALID_DRIVER_LICENSE_FIRST_YEAR_CHAR = ["3"]
VALID_DRIVER_LICENSE_LAST_YEAR_CHAR = ["1", "2"]
