"""
Module to contains architecture of LSTM
"""

import pickle5 as pickle

import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from halo import Halo

from preprocessing_pgp.name.const import GENDER_MODEL_PATH
from preprocessing_pgp.utils import suppress_warnings

suppress_warnings()


class GenderModel:
    def __init__(
        self,
        tokenizer_path: str,
        model_path: str = None
    ):
        self.tokenizer = self.__load_tokenizer(tokenizer_path)
        self.max_len = 5
        self.embed_size = 128
        self.model = self.__load_model(model_path)

    def __load_tokenizer(
        self,
        tokenizer_path: str
    ):
        with open(tokenizer_path, 'rb') as handle:
            tokenizer = pickle.load(handle)

        return tokenizer

    def __load_model(
        self,
        model_path: str = None
    ):
        if model_path is None:
            return

        model = self.build_model()
        model.load_weights(model_path)
        return model

    def build_model(self):
        inputs = tf.keras.layers.Input(shape=(self.max_len, ), name='Input')
        embed_inputs = tf.keras.layers.Embedding(len(self.tokenizer.word_index) + 1,
                                                 self.embed_size,
                                                 name='Embedding')(inputs)
        # Main architecture
        x = tf.keras.layers.LSTM(
            units=32, name='LSTM', return_sequences=True)(embed_inputs)
        x = tf.keras.layers.Dropout(0.5, name='LSTM_dropout')(x)
        x = tf.keras.layers.GlobalMaxPool1D(name='global_max_pool')(x)
        x = tf.keras.layers.Dropout(0.4, name='max_pool_dropout')(x)
        x = tf.keras.layers.Dense(4, activation='relu', name='dense')(x)
        x = tf.keras.layers.Dropout(0.3, name='dense_dropout')(x)

        outputs = tf.keras.layers.Dense(
            1, activation='sigmoid', name='Output')(x)

        model = tf.keras.Model(
            inputs=inputs, outputs=outputs, name='Accented_Model')

        model.compile(
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            metrics=['accuracy']
        )

        return model

    def predict_gender(
        self,
        names
    ):
        if self.model is None:
            return None
        encode_names = self.tokenizer.texts_to_sequences(names)
        padded_encode_names =\
            pad_sequences(
                encode_names,
                maxlen=self.max_len,
                padding='post'
            )

        genders = tf.cast(
            self.model.predict(padded_encode_names, verbose=0) > 0.5,
            dtype=tf.float32
        )
        return genders


@Halo(
    text='Predicting Genders',
    color='cyan',
    spinner='dots7',
    text_color='magenta'
)
def predict_gender_from_name(
    data: pd.DataFrame,
    name_col: str = 'name'
) -> pd.DataFrame:
    """
    Load model and predict gender from name data

    Parameters
    ----------
    data : pd.DataFrame
        The data contains customer name records
    name_col : str, optional
        The column name of the data that hold name records, by default 'name'

    Returns
    -------
    pd.DataFrame
        Data with additional columns:
        * `gender_predict`: Gender predicted from input names
    """

    gender_model = GenderModel(
        f'{GENDER_MODEL_PATH}/lstm/tokenizer.pkl',
        f'{GENDER_MODEL_PATH}/lstm/gender_lstm.h5'
    )

    data['gender_predict'] = gender_model.predict_gender(
        data[name_col].values
    )
    data['gender_predict'] = data['gender_predict'].map({
        0: 'F',
        1: 'M'
    })

    return data
