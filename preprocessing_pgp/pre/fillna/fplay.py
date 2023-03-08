
import sys
sys.path.append('/bigdata/fdp/cdp/cdp_pages/scripts_hdfs/modules/')
from helper import *
from utils import *

if __name__ == '__main__':
    
    date = datetime.strptime(sys.argv[1], '%Y-%m-%d')
    
    spark = initial_spark(mode = 'yarn', driver_memory='10g', max_worker=5)
    
    # path
    INPUT_PATH = ROOT_PATH + '/crossfill/profile_fplay.parquet'
    SAVE_PATH = ROOT_PATH + '/fillna/profile_fplay.parquet'
    
    # pipeline fillna 
    fillna_pipeline(spark, date, input_path=INPUT_PATH, save_path=SAVE_PATH, id_cttv='user_id_fplay')
    
    # post processing
    post_processing(spark, date, path=SAVE_PATH, cttv='FPLAY')
    
    spark.stop()
