import os
import argparse
import json
from time import time
from typing import List, Tuple

import pandas as pd
from tensorflow import keras
from tqdm import tqdm

try:
    from name_processing import NameProcessor
    from train_script.models.transformers import TransformerModel
    from train_script.preprocess.extract_human import HumanNameExtractor
    from preprocess import preprocess_df
except ImportError:
    from .name_processing import NameProcessor
    from .train_script.models.transformers import TransformerModel
    from .train_script.preprocess.extract_human import HumanNameExtractor
    from .preprocess import preprocess_df

tqdm.pandas()

PROJECT_PATH = '/bigdata/fdp/cdp/cdp_pages/scripts_hdfs/pre/utils/fill_accent_name'

class EnrichName:
    """
    Wrap-up module to enrich and filling accent to names
    """

    def __init__(
        self,
        model_weight_path: str,
        vectorization_paths: Tuple[str, str],
        model_config_path: str,
        split_data_path: str,
        name_rb_pth: str
    ) -> None:
        start_time = time()
        self.model = self.load_model(
            model_weight_path,
            vectorization_paths,
            model_config_path
        )
        self.fname_rb = f'{name_rb_pth}/firstname_dict.parquet'
        self.mname_rb = f'{name_rb_pth}/middlename_dict.parquet'
        self.lname_rb = f'{name_rb_pth}/lastname_dict.parquet'
        self.name_processor = NameProcessor(
            self.model,
            self.fname_rb,
            self.mname_rb,
            self.lname_rb,
            split_data_path
        )
        self.human_extractor = HumanNameExtractor('vi')
        self.total_load_time = time() - start_time

    def load_model(
        self,
        model_weight_path: str,
        vectorization_paths: Tuple[str, str],
        model_config_path: str
    ) -> TransformerModel:
        start_time = time()
        # ? Load config dict
        with open(model_config_path) as json_file:
            config_dict = json.load(json_file)

        # ? BUILD & LOAD MODEL
        source_vec_pth, target_vec_pth = vectorization_paths
        transformer = TransformerModel(
            source_vectorization=source_vec_pth,
            target_vectorization=target_vec_pth,
            config_dict=config_dict
        )
        transformer.build_model(
            optimizer=keras.optimizers.Adam(
                learning_rate=config_dict['LEARNING_RATE']),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        transformer.load_model_weights(model_weight_path)

        self.model_load_time = time() - start_time

        return transformer

    def refill_accent(
        self,
        name_df: pd.DataFrame,
        name_col: str
    ) -> pd.DataFrame:
        return self.name_processor.fill_accent(name_df, name_col)

    def unify(
        self,
        name_df: pd.DataFrame,
        name_col: str,
        key_col: str,
        keep_cols: List[str]
    ) -> pd.DataFrame:
        return self.name_processor.unify_name(
            name_df,
            name_col,
            key_col,
            keep_cols
        )

    def enrich(
        self,
        raw_df: pd.DataFrame,
        name_col: str,
        key_col: str,
        predict_first: bool = False,
        extract_human: bool = False
    ) -> pd.DataFrame:

        # ? Preprocess & Remove all non_human names
        # *-- Don't use Multi-processing cause without progress bar
        start_time = time()
        human_names, non_human_names = preprocess_df(
            raw_df,
            self.human_extractor,
            name_col=name_col,
            extract_human=extract_human
        )
        self.preprocess_time = time() - start_time

        human_names['is_human'] = True
        non_human_names['is_human'] = False

        # ? Performing enrich names & filling accent
        # * Out df with columns: ['id_group', 'raw_name', 'enrich_name', 'final_name', 'similarity_score', 'is_human']
        start_time = time()
        if predict_first:
            # Apply filling accent
            predict_df = self.refill_accent(human_names, name_col)

            # Apply unifying name
            unified_df = self.unify(
                predict_df,
                'final',
                key_col,
                keep_cols=['is_human']
            )

            # Modifying result columns
            unified_df[name_col] = human_names[name_col]
            unified_df = unified_df.rename(columns={'final': 'pred_name'})
            unified_df = unified_df[[key_col, name_col,
                                    'pred_name', 'best_name', 'similarity_score', 'is_human']]

            #! DO SOME EVALUATION HERE IF NEEDED
            #! END EVALUATION

            unified_df = unified_df.rename(
                columns={'best_name': 'final_name'})
            unified_df = unified_df[[key_col, name_col,
                                     'final_name', 'similarity_score', 'is_human']]
        else:
            # Apply unifying name
            unified_df = self.unify(
                human_names,
                name_col,
                key_col,
                keep_cols=['is_human']
            )

            # Apply filling accent
            unified_df = self.refill_accent(unified_df, 'best_name')

            # Modifying result columns
            unified_df[name_col] = human_names[name_col]
            unified_df = unified_df.rename(columns={'final': 'pred_best_name'})
            unified_df = unified_df[[key_col, name_col,
                                    'best_name', 'pred_best_name', 'similarity_score', 'is_human']]

            #! DO SOME EVALUATION HERE IF NEEDED
            #! END EVALUATION

            unified_df = unified_df.rename(
                columns={'pred_best_name': 'final_name'})
            unified_df = unified_df[[key_col, name_col,
                                     'final_name', 'similarity_score', 'is_human']]
        self.enrich_time = time() - start_time

        # ? Process for non-human names
        # * Actually not doing anything -- Just keep the same column for easy adding
        non_unified_df = non_human_names.copy(deep=True)  # (key_col, name_col)
        non_unified_df['final_name'] = non_unified_df[name_col]
        non_unified_df['similarity_score'] = 1.0

        # ? Making the final df

        final_df = pd.concat([unified_df, non_unified_df],
                             ignore_index=True)

        return final_df

    def get_time_report(self) -> pd.DataFrame:
        return pd.DataFrame({
            'model load time': [self.model_load_time],
            'total load time': [self.total_load_time],
            'preprocess time': [self.preprocess_time],
            'enrich time': [self.enrich_time]
        })


def fill_accent(
    raw_df: pd.DataFrame,
    name_col: str,
    model_pth: str = f'{PROJECT_PATH}/trial-18',
    rb_pth: str = f'{PROJECT_PATH}/trial-18',
    base_pth: str = f'{PROJECT_PATH}/name_split',
    predict_first: bool = False,
    extract_human: bool = False
) -> Tuple[pd.DataFrame, EnrichName]:
    model_weight_path = f'{model_pth}/best_transformer_model.h5'
    vectorization_paths = (
        f'{model_pth}/vecs/source_vectorization_layer.pkl',
        f'{model_pth}/vecs/target_vectorization_layer.pkl'
    )
    model_config_path = f'{model_pth}/hp.json'

    enricher = EnrichName(
        model_weight_path = model_weight_path,
        vectorization_paths=vectorization_paths,
        model_config_path=model_config_path,
        split_data_path=base_pth,
        name_rb_pth=rb_pth
    )
    
    human_names, non_human_names = preprocess_df(
        raw_df,
        enricher.human_extractor,
        name_col=name_col,
        extract_human=extract_human
    )

    human_names['is_human'] = True
    non_human_names['is_human'] = False

    # Apply filling accent
    predict_human_names = enricher.refill_accent(human_names, name_col)
    predict_human_names = predict_human_names.drop(columns=['predict']).rename(columns={'final': 'predict_name'})
    
    # Result
    result_names = pd.concat([predict_human_names, non_human_names], ignore_index=True)
    result_names.loc[result_names['predict_name'].isna(), 'predict_name'] = result_names[name_col]
    
    return result_names

def process_enrich(
    raw_df: pd.DataFrame,
    name_col: str,
    key_col: str,
    model_pth: str = f'{PROJECT_PATH}/trial-18',
    rb_pth: str = f'{PROJECT_PATH}/trial-18',
    base_pth: str = f'{PROJECT_PATH}/name_split',
    predict_first: bool = False,
    extract_human: bool = False
) -> Tuple[pd.DataFrame, EnrichName]:
    model_weight_path = f'{model_pth}/best_transformer_model.h5'
    vectorization_paths = (
        f'{model_pth}/vecs/source_vectorization_layer.pkl',
        f'{model_pth}/vecs/target_vectorization_layer.pkl'
    )
    model_config_path = f'{model_pth}/hp.json'

    enricher = EnrichName(
        model_weight_path=model_weight_path,
        vectorization_paths=vectorization_paths,
        model_config_path=model_config_path,
        split_data_path=base_pth,
        name_rb_pth=rb_pth
    )

    final_df = enricher.enrich(
        raw_df,
        name_col,
        key_col,
        predict_first=predict_first,
        extract_human=extract_human
    )

    return final_df, enricher

if __name__ == '__main__':
    # Initialize
    parser = argparse.ArgumentParser(
        prog='Enrich & Fill Accent',
        description='Script to enrich and fill accent to names'
    )

    # Adding args
    parser.add_argument('-df_pth', '--raw_df_path', type=str,
                        required=True,
                        help='The required path to raw name data in .parquet ext')
    parser.add_argument('--name_col', type=str,
                        required=True,
                        help='Required column name in DF that contains name')
    parser.add_argument('--key_col', type=str,
                        required=True,
                        help='Required column of key (phone|email|token) in DF')
    parser.add_argument('-model_pth', '--model_trial_path', type=str,
                        required=True,
                        help='The path to the model trial')
    parser.add_argument('-rb_pth', '--rule_base_path', type=str,
                        required=True,
                        help='Required path to rule based dictionary')
    parser.add_argument('-b_pth', '--base_path', type=str,
                        required=True,
                        help='Required path to name split data')
    parser.add_argument('--predict_first', action='store_true',
                        help='Whether to predict first then unify or vise-versa',
                        default=False)
    parser.add_argument('--extract_human', action='store_true',
                        help='Whether to extract human-name using NER model',
                        default=False)
    parser.add_argument('-save_pth', '--result_save_path', type=str,
                        required=True,
                        help='Required path to save the processed file')
    parser.add_argument('--device', type=str, default='0',
                        help='Define the number of GPU device (default to 0)')

    # * Parse arguments
    args = parser.parse_args()
    raw_df_path = args.raw_df_path
    name_col = args.name_col
    key_col = args.key_col
    model_trial_path = args.model_trial_path
    rule_base_path = args.rule_base_path
    base_path = args.base_path
    predict_first = args.predict_first
    extract_human = args.extract_human
    result_save_path = args.result_save_path
    device = args.device

    # * Prepare
    os.environ['CUDA_VISIBLE_DEVICES'] = device

    # * Passing to processor
    raw_df = pd.read_parquet(raw_df_path)

    final_df, enricher = process_enrich(
        raw_df,
        name_col,
        key_col,
        model_pth=model_trial_path,
        rb_pth=rule_base_path,
        base_pth=base_path,
        predict_first=predict_first,
        extract_human=extract_human
    )

    # Saving result
    print(f'\n\nSaving results to {result_save_path}...')
    if predict_first:
        final_df.to_parquet(f'{result_save_path}/enrich_predict.parquet')
    else:
        final_df.to_parquet(f'{result_save_path}/enrich_unify.parquet')

    time_report_df = enricher.get_time_report()
    time_report_df.to_parquet(f'{result_save_path}/time_report.parquet')
