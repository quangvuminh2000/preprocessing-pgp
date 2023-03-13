
"""
Module to choose best name based on email
"""

import sys

sys.path.append('/bigdata/fdp/cdp/source/core_profile/preprocess/choose_best_name')
import pipeline
sys.path.append('/bigdata/fdp/cdp/source/core_profile/preprocess/utils')
from const import (
    UTILS_PATH,
    hdfs
)

# MAIN
if __name__ == '__main__':
    # params
    DAY = sys.argv[1]

    # run
    data = pipeline.PipelineBestName(DAY, key='email', n_cores=20)

    # save
    data.to_parquet(
        f'{UTILS_PATH}/name_by_email_latest.parquet',
        filesystem=hdfs,
        index=False
    )
