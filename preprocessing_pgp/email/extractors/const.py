"""
Module contains constants for email information extraction
"""

import os

import pandas as pd

# ? STUDENT FULLNAME
fullname_path = os.path.join(
    os.path.dirname(__file__),
    '../../data/email_info/student_names.parquet'
)

FULLNAME_DICT = pd.read_parquet(fullname_path)

# ? YOB
YEAR_REGEX_DICT = {
    'full_year': '(19[5-9][0-9]|200[0-9])',
    'half_year': '([7-9][0-9])',
    'full_month': '(0[1-9]|1[012])',
    'half_month': '([1-9])',
    'full_day': '(0[1-9]|[12][0-9]|3[01])',
    'half_day': '([1-9])'
}

YEAR_FORMAT_DICT = {
    'full_year': 'yyyy',
    'half_year': 'yy',
    'full_month': 'mm',
    'half_month': 'm',
    'full_day': 'dd',
    'half_day': 'd'
}
