import os
import subprocess
from pyarrow import fs

os.environ["HTTP_PROXY"] = "http://proxy.hcm.fpt.vn:80/"
os.environ["HTTPS_PROXY"] = "http://proxy.hcm.fpt.vn:80/"
os.environ["HADOOP_CONF_DIR"] = "/etc/hadoop/conf/"
os.environ["JAVA_HOME"] = "/usr/jdk64/jdk1.8.0_112"
os.environ["HADOOP_HOME"] = "/usr/hdp/3.1.0.0-78/hadoop"
os.environ["ARROW_LIBHDFS_DIR"] = "/usr/hdp/3.1.0.0-78/usr/lib/"
os.environ["CLASSPATH"] = subprocess.check_output(
    "$HADOOP_HOME/bin/hadoop classpath --glob", shell=True
).decode("utf-8")
hdfs = fs.HadoopFileSystem(host="hdfs://hdfs-cluster.datalake.bigdata.local", port=8020)

HDFS_BASE_PTH = (
    "/data/fpt/ftel/cads/dep_solution/sa/cdp/core_profile/libs/preprocessing_pgp/data"
)

N_PROCESSES = os.cpu_count() // 2

DICT_TRASH_STRING = {
    "": None,
    "Nan": None,
    "nan": None,
    "None": None,
    "none": None,
    "Null": None,
    "null": None,
    '""': None,
}
