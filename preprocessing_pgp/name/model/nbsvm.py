"""
Module to contains architecture of Naive Bayes SVM -- Deprecated
"""

import string

import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.linear_model import LogisticRegression
from scipy import sparse

from preprocessing_pgp.name.const import GENDER_MODEL_PATH

class NBSVMClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, **logistic_kws):
        self.logistic_kws = logistic_kws

    def predict(self, x):
        # Verify that model has been fit
        check_is_fitted(self, ['_r', '_clf'])
        return self._clf.predict(x.multiply(self._r))

    def predict_proba(self, x):
        # Verify that model has been fit
        check_is_fitted(self, ['_r', '_clf'])
        return self._clf.predict_proba(x.multiply(self._r))

    def fit(self, x, y):
        # Check that X and y have correct shape
        x, y = check_X_y(x, y, accept_sparse=True)

        def pr(x, y_i, y):
            p = x[y==y_i].sum(0)
            return (p+1) / ((y==y_i).sum()+1)

        self._r = sparse.csr_matrix(np.log(pr(x,1,y) / pr(x,0,y)))
        x_nb = x.multiply(self._r)

        self._clf = LogisticRegression(**self.logistic_kws).fit(x_nb, y)
        return self

def load_model(file_path):
    with open(file_path, 'rb') as pkl_model:
        model = pickle.load(pkl_model)
    return model

def preprocess_fromName(df, name_col='Name'):
    df = df.copy()
    df['CleanName'] = df[name_col]\
        .str.lower()\
        .str.replace(r'\d', ' ', regex=True)\
        .str.replace(rf'[{string.punctuation}]', ' ', regex=True)\
        .str.replace(r'\s+', ' ', regex=True)\
        .str.strip()
    return df

def get_gender_from_accent_name(df, name_col='Name'):
    prep_df = preprocess_fromName(df.copy(), name_col)
    pipeline = load_model(f'{GENDER_MODEL_PATH}/accented/nbsvm_pipeline_v3.pkl')
    predictions = pipeline.predict(prep_df['CleanName'].values)
    predictions = list(map(lambda x: 'M' if x == 1 else 'F', predictions))
    return predictions

def get_gender_from_non_accent_name(df, name_col='Name'):
    prep_df = preprocess_fromName(df.copy(), name_col)
    pipeline = load_model(f'{GENDER_MODEL_PATH}/non_accented/nbsvm_pipeline_v3.pkl')
    predictions = pipeline.predict(prep_df['CleanName'].values)
    predictions = list(map(lambda x: 'M' if x == 1 else 'F', predictions))
    return predictions
