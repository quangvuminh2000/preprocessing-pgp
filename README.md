
# preprocessing_pgp

[![PyPI](https://shields.io/pypi/v/preprocessing-pgp)](https://pypi.org/project/nlp-preprocessing-qvm9/)
[![Python](https://img.shields.io/pypi/pyversions/preprocessing-pgp.svg?style=plastic)](https://badge.fury.io/py/preprocessing-pgp)

**preprocessing_pgp** -- The Preprocessing library for any kind of data -- is a suit of *open source Python modules, preprocessing techniques* supporting research and development in Machine Learning. preprocessing_pgp requires Python version **3.6, 3.7, 3.8, 3.9, 3.10**

## Installation

To install the current release:

```shell
pip install preprocessing-pgp
```

## Example

### 1. Preprocessing Name

```shell
python
```

```python
>>> import preprocessing_pgp as pgp
>>> pgp.preprocess.basic_preprocess_name('Phan Thị    Thúy    Hằng *$%!@#')
Phan Thị Thúy Hằng
```

### 1. Extracting Phones

```shell
python
```

```python
>>> import pandas as pd
>>> from preprocessing_pgp.phone.extractor import extract_valid_phone
>>> data = pd.read_parquet('/path/to/data.parquet')
>>> extract_valid_phone(phones=data, phone_col='col_contains_phone')
# OF PHONE CLEAN : 0

Sample of non-clean phones:
Empty DataFrame
Columns: [id, phone, clean_phone]
Index: []

100%|██████████| ####/#### [00:00<00:00, ####it/s]

# OF PHONE 10 NUM VALID : ####


# OF PHONE 11 NUM VALID : ####


0it [00:00, ?it/s]

# OF OLD PHONE CONVERTED : ####


# OF OLD REGION PHONE : ####

100%|██████████| ####/#### [00:00<00:00, ####it/s]

# OF VALID PHONE : ####

# OF INVALID PHONE : ####

Sample of invalid phones:
+------+---------+---------------+------------------+-----------+---------------+---------------+-----------------+-------------------+-------------------+
|      |      id |   clean_phone | is_phone_valid   | is_mobi   | is_new_mobi   | is_old_mobi   | phone_convert   | is_new_landline   | is_old_landline   |
+======+=========+===============+==================+===========+===============+===============+=================+===================+===================+
|   ## | 12##### |     083###### | False            | False     | False         | False         |                 | False             | False             |
+------+---------+---------------+------------------+-----------+---------------+---------------+-----------------+-------------------+-------------------+
|   ## | 12##### |     098###### | False            | False     | False         | False         |                 | False             | False             |
+------+---------+---------------+------------------+-----------+---------------+---------------+-----------------+-------------------+-------------------+
|   ## | 13##### |   039######## | False            | False     | False         | False         |                 | False             | False             |
+------+---------+---------------+------------------+-----------+---------------+---------------+-----------------+-------------------+-------------------+
|   ## | 13##### |   093######## | False            | False     | False         | False         |                 | False             | False             |
+------+---------+---------------+------------------+-----------+---------------+---------------+-----------------+-------------------+-------------------+
|   ## | 14##### |   096######## | False            | False     | False         | False         |                 | False             | False             |
+------+---------+---------------+------------------+-----------+---------------+---------------+-----------------+-------------------+-------------------+
|   ## | 14##### |   097######## | False            | False     | False         | False         |                 | False             | False             |
+------+---------+---------------+------------------+-----------+---------------+---------------+-----------------+-------------------+-------------------+
|   ## | 15##### |   098######## | False            | False     | False         | False         |                 | False             | False             |
+------+---------+---------------+------------------+-----------+---------------+---------------+-----------------+-------------------+-------------------+
|   ## | 13##### |   032######## | False            | False     | False         | False         |                 | False             | False             |
+------+---------+---------------+------------------+-----------+---------------+---------------+-----------------+-------------------+-------------------+
|   ## | 14##### |   086######## | False            | False     | False         | False         |                 | False             | False             |
+------+---------+---------------+------------------+-----------+---------------+---------------+-----------------+-------------------+-------------------+
|   ## | 14##### |   078######## | False            | False     | False         | False         |                 | False             | False             |
+------+---------+---------------+------------------+-----------+---------------+---------------+-----------------+-------------------+-------------------+
```
