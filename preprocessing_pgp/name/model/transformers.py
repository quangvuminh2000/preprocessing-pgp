from multiprocessing import Pool

from tqdm import tqdm
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import load_model

from preprocessing_pgp.name.vector_creation import load_vectorization_from_disk


# Positional Embedding for Transformer
class PositionalEmbedding(layers.Layer):
    def __init__(self, sequence_length, input_dim, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.token_embeddings = layers.Embedding(
            input_dim=input_dim, output_dim=output_dim)
        self.position_embeddings = layers.Embedding(
            input_dim=sequence_length, output_dim=output_dim)
        self.sequence_length = sequence_length
        self.input_dim = input_dim
        self.output_dim = output_dim

    def call(self, inputs):
        length = tf.shape(inputs)[-1]
        positions = tf.range(start=0, limit=length, delta=1)
        embedded_tokens = self.token_embeddings(inputs)
        embedded_positions = self.position_embeddings(positions)
        return embedded_tokens + embedded_positions

    def compute_mask(self, inputs, mask=None):
        return tf.math.not_equal(inputs, 0)

    def get_config(self):
        config = super(PositionalEmbedding, self).get_config()
        config.update({
            'output_dim': self.output_dim,
            'sequence_length': self.sequence_length,
            'input_dim': self.input_dim
        })

        return config


# Encoder model
class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, dropout=0.5, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim)
        if dropout > 0:
            self.dense_proj = tf.keras.Sequential(
                [layers.Dense(dense_dim, activation="relu"),
                 layers.Dropout(dropout),
                 layers.Dense(embed_dim), ]
            )
            self.dropout = layers.Dropout(dropout)
        else:
            self.dense_proj = tf.keras.Sequential(
                [layers.Dense(dense_dim, activation='relu'),
                 layers.Dense(embed_dim), ]
            )

        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()

    def call(self, inputs, mask=None):
        if mask is not None:
            mask = mask[:, tf.newaxis, :]
        attention_output = self.attention(
            inputs, inputs, attention_mask=mask)
        if hasattr(self, 'dropout') and self.dropout is not None:
            attention_output = self.dropout(attention_output)
        proj_input = self.layernorm_1(inputs + attention_output)
        proj_output = self.dense_proj(proj_input)
        return self.layernorm_2(proj_input + proj_output)

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "dense_dim": self.dense_dim,
        })
        return config


# Decoder Model
class TransformerDecoder(layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, dropout=0.5, **kwargs):
        super().__init__(**kwargs)

        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.attention_1 = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim)
        self.attention_2 = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim)
        if dropout > 0:
            self.dense_proj = tf.keras.Sequential(
                [layers.Dense(dense_dim, activation="relu"),
                 layers.Dropout(dropout),
                 layers.Dense(embed_dim), ]
            )
            self.dropout = layers.Dropout(dropout)
        else:
            self.dense_proj = tf.keras.Sequential(
                [layers.Dense(dense_dim, activation='relu'),
                 layers.Dense(embed_dim), ]
            )

        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.layernorm_3 = layers.LayerNormalization()
        self.supports_masking = True

    def get_config(self):
        config = super().get_config()
        config.update({
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            'dense_dim': self.dense_dim
        })

        return config

    def get_causal_attention_mask(self, inputs):
        input_shape = tf.shape(inputs)
        batch_size, sequence_length = input_shape[0], input_shape[1]
        i = tf.range(sequence_length)[:, tf.newaxis]
        j = tf.range(sequence_length)
        mask = tf.cast(i >= j, dtype='int32')
        mask = tf.reshape(mask, (1, sequence_length, sequence_length))
        mult = tf.concat(
            [tf.expand_dims(batch_size, -1),
             tf.constant([1, 1], dtype=tf.int32)],
            axis=0
        )

        return tf.tile(mask, mult)

    def call(self, inputs, encoder_outputs, mask=None):
        causal_mask = self.get_causal_attention_mask(inputs)

        if mask is not None:
            padding_mask = tf.cast(mask[:, tf.newaxis, :], dtype='int32')
            padding_mask = tf.minimum(padding_mask, causal_mask)

        attention_output_1 = self.attention_1(
            query=inputs,
            value=inputs,
            key=inputs,
            attention_mask=causal_mask)
        if hasattr(self, 'dropout') and self.dropout is not None:
            attention_output_1 = self.dropout(attention_output_1)
        attention_output_1 = self.layernorm_1(inputs + attention_output_1)
        attention_output_2 = self.attention_2(
            query=attention_output_1,
            value=encoder_outputs,
            key=encoder_outputs,
            attention_mask=padding_mask,
        )
        if hasattr(self, 'dropout') and self.dropout is not None:
            attention_output_2 = self.dropout(attention_output_2)
        attention_output_2 = self.layernorm_2(
            attention_output_1 + attention_output_2)
        proj_output = self.dense_proj(attention_output_2)
        return self.layernorm_3(attention_output_2 + proj_output)


# Entire Transformer model
class TransformerModel:
    def __init__(self, source_vectorization, target_vectorization,
                 config_dict,
                 model_path=None, weight_path=None):

        self.source_vectorization = self.load_vectorization(
            source_vectorization)
        self.target_vectorization = self.load_vectorization(
            target_vectorization)

        self.load_model(model_path)
        self.load_model_weights(weight_path)

        self.sequence_length = config_dict['SEQUENCE_LENGTH']
        self.vocab_size = config_dict['VOCAB_SIZE']
        self.embed_dim = config_dict['EMBED_DIM']
        self.dense_dim = config_dict['DENSE_DIM']
        self.num_heads = config_dict['NUM_HEADS']
        self.drop_out = config_dict['DROPOUT_RATE']
        self.drop_out_enc = config_dict['DROPOUT_ENC']
        self.drop_out_dec = config_dict['DROPOUT_DEC']

    def load_vectorization(self, vectorization):
        if isinstance(vectorization, str):
            return load_vectorization_from_disk(vectorization)
        else:
            return vectorization

    def load_model(self, model_path: str = None):
        if model_path is None:
            return

        self.model = load_model(
            model_path,
            custom_objects={
                'PositionalEmbedding': PositionalEmbedding,
                'TransformerDecoder': TransformerDecoder,
                'TransformerEncoder': TransformerEncoder
            }
        )

    def load_model_weights(self, weight_path: str = None):
        if weight_path is None:
            return

        if not hasattr(self, 'model') or self.model is None:
            raise TypeError(
                'Please build the model or load a pre-trained model first')

        self.model.load_weights(weight_path)

    def build_model(self, *arg, **kwargs):
        encoder_inputs = tf.keras.Input(
            shape=(None,), dtype='int64', name='stripped')
        x = PositionalEmbedding(
            self.sequence_length, self.vocab_size, self.embed_dim)(encoder_inputs)
        encoder_outputs = TransformerEncoder(
            self.embed_dim, self.dense_dim, self.num_heads, self.drop_out_enc)(x)

        decoder_inputs = tf.keras.Input(
            shape=(None,), dtype='int64', name='original')
        x = PositionalEmbedding(
            self.sequence_length, self.vocab_size, self.embed_dim)(decoder_inputs)
        x = TransformerDecoder(
            self.embed_dim, self.dense_dim, self.num_heads, self.drop_out_dec)(x, encoder_outputs)

        if self.drop_out > 0:
            x = layers.Dropout(self.drop_out)(x)

        decoder_outputs = layers.Dense(
            self.vocab_size, activation='softmax')(x)

        self.model = tf.keras.Model(
            inputs=[encoder_inputs, decoder_inputs], outputs=[decoder_outputs])

        self.model.compile(*arg, **kwargs)

    def fit(self, *arg, **kwargs):
        if not hasattr(self, 'model') or self.model is None:
            raise TypeError(
                'Please build the model or load a pre-trained model first')

        return self.model.fit(*arg, **kwargs)

    def evaluate(self, *args, **kwargs):
        if not hasattr(self, 'model') or self.model is None:
            raise TypeError(
                'Please build the model or load a pre-trained model first')

        rs = self.model.evaluate(*args, **kwargs)
        print(rs)
        return rs

    def predict(self, input_sentence):
        if not hasattr(self, 'model') or self.model is None:
            raise TypeError(
                'Please build the model or load a pre-trained model first')

        if not hasattr(self, 'original_vocal'):
            self.original_vocal = self.target_vectorization.get_vocabulary()
            self.original_index_lookup = dict(
                zip(range(len(self.original_vocal)), self.original_vocal))

        input_word = input_sentence.split()
        tokenized_input_sentence = self.source_vectorization([input_sentence])
        decoded_sentence = '[start]'
        
        for i in range(min(self.sequence_length, len(input_word))):
            tokenized_target_sentence = self.target_vectorization([decoded_sentence])[
                :, :-1]
            predictions = self.model(
                [tokenized_input_sentence, tokenized_target_sentence]).numpy()
            
            # Change to beam search here if possible
            sampled_token_index = np.argmax(predictions[0, i, :])
            sampled_token = self.original_index_lookup[
                sampled_token_index] if sampled_token_index != 1 else input_word[i]
            decoded_sentence += ' ' + sampled_token

        return decoded_sentence.replace('[start]', '').strip()
    
    def predict_multi(self, sentences: pd.Series,
                      multiprocessing: bool = False,
                      n_cpu: int = 1) -> pd.Series:
        """
        Predict multiple sentences using whether parallel or single process

        Parameters
        ----------
        sentences : pd.Series
            The input sentence series containing the names
        multiprocessing : bool, optional
            Whether using parallel execution, by default False
        n_cpu : int, optional
            The number of cpu use in parallel execution, by default 1

        Returns
        -------
        pd.Series
            The returned prediction name series
        """
        if multiprocessing:
            with Pool(processes=n_cpu) as pool:
                pred_sentences = list(
                    tqdm(pool.imap(self.predict, sentences), total=sentences.size))

        else:
            pred_sentences = []
            for sent in tqdm(sentences):
                pred_sentences.append(self.predict(sent))

        pred_result = pd.Series(pred_sentences, dtype='str', name='pred_sent')

        return pred_result

