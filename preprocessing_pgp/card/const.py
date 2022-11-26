import pandas as pd
import os


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

