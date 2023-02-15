"""
Module contains constants for email information extraction
"""

import os

import pandas as pd

# ? STUDENT FULLNAME
fullname_path = os.path.join(
    os.path.dirname(__file__),
    '../../data/email_info/norm_name_trace_dict.parquet'
)

FULLNAME_DICT = pd.read_parquet(fullname_path)

# ? YOB
dob_path = os.path.join(
    os.path.dirname(__file__),
    '../../data/email_info/dob_dict.parquet'
)
DOB_DICT = pd.read_parquet(dob_path)

DOB_REGEX_DICT = DOB_DICT.set_index('dob_type')['regex'].to_dict()
DOB_NUM_DIGIT_DICT = DOB_DICT.set_index('dob_type')['num_digit'].to_dict()
DOB_FORMAT_DICT = DOB_DICT.set_index('dob_type')['format'].to_dict()

# ? PHONE

VIETNAMESE_PHONE_REGEX = r'((02[0-9]|0|84)[3|5|7|8|9]*[0-9]{8})'
