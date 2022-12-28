import pickle

import tensorflow as tf
from tensorflow.keras import layers


def create_vectorizations(train_pairs, sequence_length=50, vocab_size=15000):
    source_vectorization = layers.TextVectorization(
        max_tokens=vocab_size,
        output_mode='int',
        output_sequence_length=sequence_length
    )
    train_stripped_texts = [n[0] for n in train_pairs]
    source_vectorization.adapt(train_stripped_texts)

    target_vectorization = layers.TextVectorization(
        max_tokens=vocab_size,
        output_mode='int',
        output_sequence_length=sequence_length+1
    )
    train_original_texts = [n[1] for n in train_pairs]
    target_vectorization.adapt(train_original_texts)

    return source_vectorization, target_vectorization


def save_vectorization(vectorization, file_path):
    '''
    Save the config and weights of a vectorization to disk as pickle file,
    so that we can reuse it when making inference.
    '''

    with open(file_path, 'wb') as f:
        f.write(pickle.dumps({'config': vectorization.get_config(),
                              'weights': vectorization.get_weights()}))


def load_vectorization_from_disk(vectorization_path):
    '''
    Load a saved vectorization from disk.
    This method is based on the following answer on Stackoverflow.
    https://stackoverflow.com/a/65225240/4510614
    '''

    with open(vectorization_path, 'rb') as f:
        from_disk = pickle.load(f)
        new_v = layers.TextVectorization(max_tokens=from_disk['config']['max_tokens'],
                                         output_mode='int',
                                         output_sequence_length=from_disk['config']['output_sequence_length'])

        # You have to call `adapt` with some dummy data (BUG in Keras)
        new_v.adapt(tf.data.Dataset.from_tensor_slices(["xyz"]))
        new_v.set_weights(from_disk['weights'])
    return new_v
