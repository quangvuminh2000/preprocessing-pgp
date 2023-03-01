"""
Module to choose best name based on phone
"""

import sys

sys.path.append('/bigdata/fdp/cdp/cdp_pages/scripts_hdfs/pre/utils/new')
import pipeline_best
from const import (
    UTILS_PATH,
    hdfs
)

# MAIN
if __name__ == '__main__':
    # params
    DAY = sys.argv[1]

    # run
    data = pipeline_best.PipelineBestName(DAY, key='phone', n_cores=24)

    # save
    data.to_parquet(
        f'{UTILS_PATH}/name_by_phone_latest_new.parquet',
        filesystem=hdfs,
        index=False
    )
