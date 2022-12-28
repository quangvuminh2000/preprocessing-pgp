import os
import json
import argparse
from string import capwords
from typing import List

import pandas as pd
from tqdm import tqdm
from unidecode import unidecode
from tensorflow import keras

try:
    from split_name import NameProcess
    from train_script.models.transformers import TransformerModel
    from train_script.postprocess.rulebase_name import rule_base_name
except ImportError:
    from .split_name import NameProcess
    from .train_script.models.transformers import TransformerModel
    from .train_script.postprocess.rulebase_name import rule_base_name

tqdm.pandas()


class NameProcessor:
    def __init__(self,
                 model: TransformerModel,
                 firstname_rb: pd.DataFrame,
                 middlename_rb: pd.DataFrame,
                 lastname_rb: pd.DataFrame,
                 base_path: str
                 ):
        self.model = model
        self.name_dicts = (firstname_rb, middlename_rb, lastname_rb)
        self.name_process = NameProcess(base_path)

    def predict_non_accent(self, name: str):
        de_name = unidecode(name)

        # Keep case already have accent
        if name != de_name:
            return name

        # Only apply to case not having accent
        return capwords(self.model.predict(name))

    def fill_accent(self,
                    name_df: pd.DataFrame,
                    name_col: str,
                    ):
        predicted_name = name_df.copy(deep=True)

        print('\n\n')
        print("Filling diacritics to names...")
        predicted_name['predict'] = predicted_name[name_col].progress_apply(
            self.predict_non_accent)

        print('\n\n')

        print("Applying rule-based postprocess...")
        predicted_name['final'] = predicted_name.progress_apply(
            lambda row: rule_base_name(
                row['predict'], unidecode(row[name_col]), self.name_dicts),
            axis=1
        )

        print('\n\n')

        return predicted_name

    def unify_name(self,
                   name_df: pd.DataFrame,
                   name_col: str,
                   key_col: str,
                   keep_cols: List[str]):
        name_df[key_col] = name_df[key_col].astype('str')
        best_name_df = self.name_process.CoreBestName(
            name_df,
            name_col=name_col,
            key_col=key_col
        )

        return best_name_df[[key_col, name_col, 'best_name', 'similarity_score', *keep_cols]]


def process_name(clean_name_path: str,
                 model_trial_path: str,
                 rule_base_path: str,
                 base_path: str,
                 name_col: str,
                 key_col: str,
                 predict_first: bool = True,
                 result_save_pth: str = 'result.parquet'
                 ):
    # Data load
    name_df = pd.read_parquet(clean_name_path)
    # Model required paths
    cfg_path = f'{model_trial_path}/hp.json'
    source_vec_path = f'{model_trial_path}/vecs/source_vectorization_layer.pkl'
    target_vec_path = f'{model_trial_path}/vecs/target_vectorization_layer.pkl'
    model_weight_path = f'{model_trial_path}/best_transformer_model.h5'

    # Loading config & Build model
    with open(cfg_path) as json_file:
        config_dict = json.load(json_file)

    transformer = TransformerModel(source_vec_path,
                                   target_vec_path,
                                   config_dict)
    transformer.build_model(optimizer=keras.optimizers.Adam(learning_rate=config_dict['LEARNING_RATE']),
                            loss='sparse_categorical_crossentropy',
                            metrics=['accuracy'])
    transformer.load_model_weights(model_weight_path)

    # Load required rule-based
    fname_dict_df = pd.read_parquet(f'{rule_base_path}/firstname_dict.parquet')
    mname_dict_df = pd.read_parquet(
        f'{rule_base_path}/middlename_dict.parquet')
    lname_dict_df = pd.read_parquet(f'{rule_base_path}/lastname_dict.parquet')

    name_processor = NameProcessor(transformer,
                                   firstname_rb=fname_dict_df,
                                   middlename_rb=mname_dict_df,
                                   lastname_rb=lname_dict_df,
                                   base_path=base_path)

    if predict_first:
        # Apply filling accent
        predict_df = name_processor.fill_accent(name_df, name_col)

        # Apply unifying name
        unified_df = name_processor.unify_name(predict_df, 'final', key_col)

        # Modifying result columns
        unified_df['raw_name'] = name_df[name_col]
        unified_df = unified_df.rename(columns={'final': 'pred_name'})
        unified_df = unified_df[[key_col, 'raw_name',
                                 'pred_name', 'best_name', 'similarity_score']]
    else:
        # Apply unifying name
        unified_df = name_processor.unify_name(name_df, name_col, key_col)

        # Apply filling accent
        unified_df = name_processor.fill_accent(unified_df, 'best_name')

        # Modifying result columns
        unified_df['raw_name'] = name_df[name_col]
        unified_df = unified_df.rename(columns={'final': 'pred_best_name'})
        unified_df = unified_df[[key_col, 'raw_name',
                                 'best_name', 'pred_best_name', 'similarity_score']]

    # Saving result
    unified_df.to_parquet(result_save_pth)


if __name__ == '__main__':
    # Initialize
    parser = argparse.ArgumentParser(
        description='Processor for filling accent and unify names')

    # Adding required args
    parser.add_argument('-cln_pth', '--clean_name_path', type=str, required=True,
                        help='A required path to the clean name data .parquet ext')
    parser.add_argument('-model_pth', '--model_trial_path', type=str, required=True,
                        help='A required path to the model trial config')
    parser.add_argument('-rb_pth', '--rule_base_path', type=str, required=True,
                        help='Required path to rule based dictionary')
    parser.add_argument('-b_pth', '--base_path', type=str, required=True,
                        help='Required path to name split data')
    parser.add_argument('--name_col', type=str, required=True,
                        help='Required column name contains name')
    parser.add_argument('--key_col', type=str, required=True,
                        help='Required column name contains key')
    parser.add_argument('--predict_first', action='store_true',
                        help='Whether to predict first then unify or vise-versa',
                        default=False)
    parser.add_argument('-save_pth', '--result_save_pth', type=str, required=True,
                        help='Required path to save the processed file')
    parser.add_argument('--device', type=str, default='0',
                        help='Define the number of GPU device (default to 0)')

    # Parse
    args = parser.parse_args()

    # Access
    clean_name_path = args.clean_name_path
    model_trial_path = args.model_trial_path
    rule_base_path = args.rule_base_path
    base_path = args.base_path
    name_col = args.name_col
    key_col = args.key_col
    predict_first = args.predict_first
    result_save_pth = args.result_save_pth
    device = args.device

    # Passing to processor
    os.environ['CUDA_VISIBLE_DEVICES'] = device

    process_name(clean_name_path=clean_name_path,
                 model_trial_path=model_trial_path,
                 rule_base_path=rule_base_path,
                 base_path=base_path,
                 name_col=name_col,
                 key_col=key_col,
                 predict_first=predict_first,
                 result_save_pth=result_save_pth)
