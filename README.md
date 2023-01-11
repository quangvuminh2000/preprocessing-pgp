
# preprocessing_pgp

[![PyPI](https://shields.io/pypi/v/preprocessing-pgp)](https://pypi.org/project/preprocessing-pgp)
[![Python](https://img.shields.io/pypi/pyversions/preprocessing-pgp.svg?style=plastic)](https://badge.fury.io/py/preprocessing-pgp)
[![License](https://img.shields.io/github/license/quangvuminh2000/preprocessing-pgp)](https://raw.githubusercontent.com/quangvuminh2000/preprocessing-pgp/main/LICENSE.txt)
[![Downloads](https://img.shields.io/pypi/dm/preprocessing-pgp?label=pypi%20downloads)](https://pepy.tech/project/preprocessing-pgp)
[![linting: pylint](https://img.shields.io/badge/linting-pylint-yellowgreen)](https://github.com/PyCQA/pylint)

**preprocessing_pgp** -- The Preprocessing library for any kind of data -- is a suit of *open source Python modules, preprocessing techniques* supporting research and development in Machine Learning. preprocessing_pgp requires Python version **3.6, 3.7, 3.8, 3.9, 3.10**

---

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

---

## Features

### 1. Vietnamese Naming Functions

#### 1.1. Preprocessing Names

```shell
python
```

```python
>>> import preprocessing_pgp.name.preprocess import basic_preprocess_name
>>> basic_preprocess_name('Phan Thị    Thúy    Hằng *$%!@#')
Phan Thị Thúy Hằng
```

#### 1.2. Enrich Vietnamese Names (New Features)

```shell
python
```

```python
>>> import pandas as pd
>>> from preprocessing_pgp.name.enrich_name import process_enrich
>>> data = pd.read_parquet('/path/to/data.parquet')
>>> enrich_data = process_enrich(data, name_col='name')


Cleansing Takes 0m0s


Enrich names takes 5m10s

>>> enrich_data.columns
Index(['name', 'predict', 'final'], dtype='object')
```

### 2. Extracting Vietnamese Phones

```shell
python
```

```python
>>> import pandas as pd
>>> from preprocessing_pgp.phone.extractor import extract_valid_phone
>>> data = pd.read_parquet('/path/to/data.parquet')
>>> extracted_data = extract_valid_phone(phones=data, phone_col='phone', print_info=True)
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

### 3. Verify Vietnamese Card IDs

```shell
python
```

```python
>>> import pandas as pd
>>> from preprocessing_pgp.card.validation import verify_card
>>> data = pd.read_parquet('/path/to/data.parquet')
>>> verified_data = verify_card(data, card_col='card_id', print_info=True)

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

### 4. Extract Information in Vietnamese Address

> All the region codes traced are retrieve from [Đơn Vị Hành Chính Việt Nam](http://tongdieutradanso.vn/don-vi-hanh-chinh-viet-nam.html)

Apart from original columns of **dataframe**, we also generate columns with specific meanings:

* **cleaned_*<address_col>*** : The *cleaned address* retrieve from the raw address column
* **level 1** : The raw city extracted from the *cleaned address*
* **best level 1** : The *beautified city* traced from extracted raw city
* **level 1 code** : The generated *city code*
* **level 2** : The raw district extracted from the *cleaned address*
* **best level 2** : The *beautified district* traced from extracted raw district
* **level 2 code** : The generated *district code*
* **level 3** : The raw ward extracted from the *cleaned address*
* **best level 3** : The *beautified ward* traced from extracted raw ward
* **level 3 code** : The generated *ward code*
* **remained address** : The *remaining address* not being extracted

```shell
python
```

```python
>>> import pandas as pd
>>> from preprocessing_pgp.address.extractor import extract_vi_address
>>> data = pd.read_parquet('/path/to/data.parquet')
>>> extracted_data = extract_vi_address(data, address_col='address')
Cleansing takes 0m0s


Extracting takes 0m22s


Code generation takes 0m3s

>>> extracted_data.columns
Index(['address', 'cleaned_address', 'level 1', 'best level 1', 'level 2',
       'best level 2', 'level 3', 'best level 3', 'remained address',
       'level 1 code', 'level 2 code', 'level 3 code'],
      dtype='object')
```

### 5. Validate email address

A valid email is consist of:

1. Large company email's address (@gmail, @yahoo, @outlook, etc.)
2. Common email address (contains at least a alphabet character in email's name)
3. Education email (can start with a number)
4. Not auto-email

Apart from original columns of **dataframe**, we also generate columns with specific meanings:

* **is_email_valid** : indicator of whether the email is valid or not

```shell
python
```

```python
>>> import pandas as pd
>>> from preprocessing_pgp.email.validator import process_validate_email
>>> data = pd.read_parquet('/path/to/data.parquet')
>>> validated_data = process_validate_email(data, email_col='email')
Cleansing email takes 0m0s


Validating email takes 0m22s
```
