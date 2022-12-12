import os

import pandas as pd
import numpy as np

from preprocessing_pgp.card.utils import digit_to_year_string


# * IMPORTANT CONSTANTS
CURRENT_YEAR = 22
OLDEST_YEAR_DRIVER_LICENSE = 45


# * PERSONAL IDENTIFICATION CARD
__OLD_PID_CODE_PATH = os.path.join(
    os.path.dirname(__file__), os.pardir, "data", "old_codes.parquet"
)
__NEW_PID_CODE_PATH = os.path.join(
    os.path.dirname(__file__), os.pardir, "data", "new_codes.parquet"
)

POSSIBLE_GENDER_NUM = ['0', '1', '2', '3']
GENDER_NUM_TO_CENTURY = {
    '20': ['0', '1'],
    '21': ['2', '3']
}
OLD_PID_REGION_CODE_NUMS = pd.read_parquet(__OLD_PID_CODE_PATH)['code'].values
NEW_PID_REGION_CODE_NUMS = pd.read_parquet(__NEW_PID_CODE_PATH)['code'].values
OLD_PID_CODE_LENGTH = 9
NEW_PID_CODE_LENGTH = 12
LIMIT_DOB_PID = 14
VALID_PID_21_CENTURY_DOB = [
    digit_to_year_string(dob)
    for dob in range(0, CURRENT_YEAR-LIMIT_DOB_PID+1)
]

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
    [digit_to_year_string(pass_year)
        for pass_year
        in range(CURRENT_YEAR+1, OLDEST_YEAR_DRIVER_LICENSE)
     ]

DRIVER_LICENSE_LENGTH = 12
INVALID_DRIVER_LICENSE_FIRST_YEAR_CHAR = ["3"]
VALID_DRIVER_LICENSE_LAST_YEAR_CHAR = ["1", "2"]
