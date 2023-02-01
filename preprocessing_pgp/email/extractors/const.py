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
