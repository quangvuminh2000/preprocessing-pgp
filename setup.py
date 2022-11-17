import pathlib
from setuptools import setup, find_packages

# ? setup() does all the work
setup(
    include_dirs=['preprocessing_pgp/data'],
    package_data={
        "preprocessing_pgp.data": ["*.parquet"]
    }
)
