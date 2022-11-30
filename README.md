
# preprocessing_pgp

[![PyPI](https://shields.io/pypi/v/preprocessing-pgp)](https://pypi.org/project/preprocessing-pgp)
[![Python](https://img.shields.io/pypi/pyversions/preprocessing-pgp.svg?style=plastic)](https://badge.fury.io/py/preprocessing-pgp)
[![License](https://img.shields.io/github/license/quangvuminh2000/preprocessing-pgp)](https://raw.githubusercontent.com/quangvuminh2000/preprocessing-pgp/main/LICENSE.txt)
[![Downloads](https://img.shields.io/pypi/dm/preprocessing-pgp?label=pypi%20downloads)](https://pepy.tech/project/preprocessing-pgp)

**preprocessing_pgp** -- The Preprocessing library for any kind of data -- is a suit of *open source Python modules, preprocessing techniques* supporting research and development in Machine Learning. preprocessing_pgp requires Python version **3.6, 3.7, 3.8, 3.9, 3.10**

## Installation

To install the **current release**:

```shell
pip install preprocessing-pgp
```

To install the release with **specific version** (e.g. 0.1.3):

```shell
pip install preprocessing-pgp==0.1.3
```

To upgrade package to **latest version**:

```shell
pip install --upgrade preprocessing-pgp
```

## Examples

### 1. Preprocessing Name

```shell
python
```

```python
>>> import preprocessing_pgp as pgp
>>> pgp.preprocess.basic_preprocess_name('Phan Thị    Thúy    Hằng *$%!@#')
Phan Thị Thúy Hằng
```

### 2. Extracting Phones

```shell
python
```

```python
>>> import pandas as pd
>>> from preprocessing_pgp.phone.extractor import extract_valid_phone
>>> data = pd.read_parquet('/path/to/data.parquet')
>>> extracted_data = extract_valid_phone(phones=data, phone_col='phone')
# OF PHONE CLEANED : 0

Sample of non-clean phones:
Empty DataFrame
Columns: [id, phone, clean_phone]
Index: []

100%|██████████| ####/#### [00:00<00:00, ####it/s]

# OF PHONE 10 NUM VALID : ####


# OF PHONE 11 NUM VALID : ####


0it [00:00, ?it/s]

# OF OLD PHONE CONVERTED : ####


# OF OLD LANDLINE PHONE : ####

100%|██████████| ####/#### [00:00<00:00, ####it/s]

# OF VALID PHONE : ####

# OF INVALID PHONE : ####

Sample of invalid phones:
+------+---------+-------------+------------------+-----------+---------------+---------------+-------------------+-------------------+-----------------+
|      |      id |       phone | is_phone_valid   | is_mobi   | is_new_mobi   | is_old_mobi   | is_new_landline   | is_old_landline   | phone_convert   |
+======+=========+=============+==================+===========+===============+===============+===================+===================+=================+
|   47 | ####### |   083###### | False            | False     | False         | False         | False             | False             |                 |
+------+---------+-------------+------------------+-----------+---------------+---------------+-------------------+-------------------+-----------------+
|  317 | ####### |   098###### | False            | False     | False         | False         | False             | False             |                 |
+------+---------+-------------+------------------+-----------+---------------+---------------+-------------------+-------------------+-----------------+
|  398 | ####### | 039######## | False            | False     | False         | False         | False             | False             |                 |
+------+---------+-------------+------------------+-----------+---------------+---------------+-------------------+-------------------+-----------------+
|  503 | ####### | 093######## | False            | False     | False         | False         | False             | False             |                 |
+------+---------+-------------+------------------+-----------+---------------+---------------+-------------------+-------------------+-----------------+
| 1261 | ####### | 096######## | False            | False     | False         | False         | False             | False             |                 |
+------+---------+-------------+------------------+-----------+---------------+---------------+-------------------+-------------------+-----------------+
| 1370 | ####### | 097######## | False            | False     | False         | False         | False             | False             |                 |
+------+---------+-------------+------------------+-----------+---------------+---------------+-------------------+-------------------+-----------------+
| 1554 | ####### | 098######## | False            | False     | False         | False         | False             | False             |                 |
+------+---------+-------------+------------------+-----------+---------------+---------------+-------------------+-------------------+-----------------+
| 2469 | ####### | 032######## | False            | False     | False         | False         | False             | False             |                 |
+------+---------+-------------+------------------+-----------+---------------+---------------+-------------------+-------------------+-----------------+
| 2609 | ####### | 086######## | False            | False     | False         | False         | False             | False             |                 |
+------+---------+-------------+------------------+-----------+---------------+---------------+-------------------+-------------------+-----------------+
| 2750 | ####### | 078######## | False            | False     | False         | False         | False             | False             |                 |
+------+---------+-------------+------------------+-----------+---------------+---------------+-------------------+-------------------+-----------------+
```

### 3. Verify Card IDs

```shell
python
```

```python
>>> import pandas as pd
>>> from preprocessing_pgp.card.validation import verify_card
>>> data = pd.read_parquet('/path/to/data.parquet')
>>> verified_data = verify_card(data, card_col='card_id')

##### CLEANSING #####


# NAN CARD ID: ####


# CARD ID CONTAINS NON-DIGIT CHARACTERS: ####


SAMPLE OF CARDS WITH NON-DIGIT CHARACTERS:
              card_id  is_valid  is_personal_id
#######      B#######     False           False
#######      C#######     False           False
#######       G######     False           False
#######     A########     False           False
#######  ###########k     False           False
#######  ###########k     False           False
#######      C#######     False           False
#######      B#######     False           False
#######  PT AR#######     False           False
#######     E########     False           False



# CARD OF LENGTH 9 OR 12: #######
STATISTIC:
True     ######
False     #####
Name: is_valid, dtype: int64




# CARD OF LENGTH 8 OR 11: ###
STATISTIC:
True     ######
False     #####
Name: is_valid, dtype: int64



# CARD WITH OTHER LENGTH: ####
# PASSPORT FOUND: ####


SAMPLE OF PASSPORT:
          card_id  is_valid  card_length clean_card_id  is_passport
#######  B#######      True            8      B#######         True
#######  C#######      True            8      C#######         True
#######  C#######      True            8      C#######         True
#######  B#######      True            8      B#######         True
#######  B#######      True            8      B#######         True
#######  B#######      True            8      B#######         True
#######  C#######      True            8      C#######         True
#######  B#######      True            8      B#######         True
#######  B#######      True            8      B#######         True
#######  B#######      True            8      B#######         True




# DRIVER LICENSE FOUND: 41461


SAMPLE OF DRIVER LICENSE:
          card_id  is_valid  is_personal_id  ...  clean_card_id is_passport  is_driver_license
47   0###########      True           False  ...   0###########       False               True
74   0###########      True           False  ...   0###########       False               True
170  0###########      True           False  ...   0###########       False               True
179  0###########      True           False  ...   0###########       False               True
206  0###########      True           False  ...   0###########       False               True
282  0###########      True           False  ...   0###########       False               True
295  0###########      True           False  ...   0###########       False               True
616  0###########      True           False  ...   0###########       False               True
663  0###########      True           False  ...   0###########       False               True
671  0###########      True           False  ...   0###########       False               True


##### GENERAL CARD ID REPORT #####

COHORT SIZE: #######
STATISTIC:
True     ######
False     #####
PASSPORT: ####
DRIVER LICENSE: ####
```

### 4. Enrich Vietnamese Names (New Features)

```shell
python
```

```python
>>> import pandas as pd
>>> from preprocessing_pgp.name.enrich_name import process_enrich
>>> data = pd.read_parquet('/path/to/data.parquet')
>>> enrich_data, _ = process_enrich(data, name_col='name')
Basic pre-processing names...
100%|████████████████████████████████████| 1000/1000 [00:00<00:00, 19669.68it/s]



--------------------
0 names have been clean!
--------------------




Filling diacritics to names...
100%|███████████████████████████████████████| 1000/1000 [01:29<00:00, 11.23it/s]

AVG prediction time : 0.0890703010559082s



Applying rule-based postprocess...
100%|████████████████████████████████████| 1000/1000 [00:00<00:00, 38292.26it/s]

AVG rb time : 2.671933174133301e-05s


>>> enrich_data.columns
Index(['name', 'predict', 'final'], dtype='object')
```
