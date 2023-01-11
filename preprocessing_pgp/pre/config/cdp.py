import sys
sys.path.insert(1, '/bigdata/fdp/cdp/script/')

import os
import pandas as pd
import numpy as np
import re

###############################################################################

from config.cfg import Config
from raw_profile.profile_validate import validate_phone_email
from raw_profile.sensitive import hide_sensitive_columns
from helper.multiprocessing import read_all
from helper.segmentizer import Segmentizer
from helper.spark import *
from helper.identity_resolver import IdentityResolver
# from helper.cookie_and_ip import get_cookie
# from helper.sensitive import decrypt

config = Config()

SAVE_PATH = '/bigdata/fdp/cdp/profile/rawdata/'

def get_last_day(path):
    last_file = sorted([
        file for file in os.listdir(path) if os.path.isfile(os.path.join(path, file)) and re.search('[0-9]', file)
    ])[-1]
    last_day = re.search(r'(20[0-9]{2}-[0-9]{2}-[0-9]{2})', last_file).group(1)
    return last_day

def get_last_file_hdfs(path):
    return sorted([f.path for f in hdfs.get_file_info(fs.FileSelector(path))])[-1]

def handle_error_date(x):
    x[x < '1900-01-01'] = np.nan
    return x
