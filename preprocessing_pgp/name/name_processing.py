from string import capwords
from typing import List

import pandas as pd
from preprocessing_pgp.name.const import WITH_ACCENT_ELEMENTS
from preprocessing_pgp.name.model.transformers import TransformerModel
from preprocessing_pgp.name.preprocess import (
    remove_duplicated_name,
    remove_invalid_base_element,
    remove_invalid_element,
)
from preprocessing_pgp.name.rulebase_name import rule_base_name
from preprocessing_pgp.name.split_name import NameProcess
from preprocessing_pgp.name.utils import remove_nicknames
from preprocessing_pgp.utils import replace_trash_string
from tqdm import tqdm
from unidecode import unidecode

tqdm.pandas()


class NameProcessor:
    def __init__(
        self,
        model: TransformerModel,
        firstname_rb: pd.DataFrame,
        middlename_rb: pd.DataFrame,
        lastname_rb: pd.DataFrame,
    ):
        self.model = model
        self.name_dicts = (firstname_rb, middlename_rb, lastname_rb)
        self.name_process = NameProcess()

    def __is_valid_element(
        self, name_part: str, base_elements: set = None
    ) -> bool:
        if name_part is None:
            return None
        return name_part.lower() in base_elements

    def choose_better_enrich(self, raw_name: str, enrich_name: str) -> str:
        """
        Decide whether the best part to choose from enrich name
        """
        if raw_name is None or enrich_name is None:
            return None
        raw_components = raw_name.split(" ")
        enrich_components = enrich_name.split(" ")

        # * Enrich wrong
        if len(raw_components) != len(enrich_components):
            return None

        best_name_components = []
        for i, component in enumerate(raw_components):
            if self.__is_valid_element(component, WITH_ACCENT_ELEMENTS):
                best_name_components.append(component)
            elif self.__is_valid_element(
                enrich_components[i], WITH_ACCENT_ELEMENTS
            ):
                best_name_components.append(enrich_components[i])
            else:
                continue

            # if is_name_accented(enrich_components[i]) and not is_name_accented(component):
            #     if self.__is_valid_element(component, WITH_ACCENT_ELEMENTS):
            #         best_name_components.append(component)
            #     elif self.__is_valid_element(enrich_components[i], WITH_ACCENT_ELEMENTS):
            #         best_name_components.append(enrich_components[i])
            #     else:
            #         # best_name_components.append(unidecode(component))
            #         continue
            # else:
            #     if self.__is_valid_element(component, WITH_ACCENT_ELEMENTS):
            #         best_name_components.append(component)
            #     else:
            #         # best_name_components.append(unidecode(component))
            #         continue

        return " ".join(best_name_components).strip()

    def predict_non_accent(self, name: str):
        if name is None:
            return None

        # # * Skip process for case with valid accent
        # if np.intersect1d(
        #         name.split(),
        #         WITH_ACCENT_ELEMENTS
        #     ).shape[0] == len(name.split()):
        #     return name
        # Keep case already have accent
        # if name != de_name:
        #     return name

        # Predict name
        de_name = unidecode(name)
        predicted_name = capwords(self.model.predict(de_name))

        return predicted_name

    def __is_good_name(self, name: str) -> bool:
        return all(
            [
                self.__is_valid_element(part, WITH_ACCENT_ELEMENTS)
                for part in name.split()
            ]
        )

    def fill_accent(
        self,
        name_df: pd.DataFrame,
        name_col: str,
    ):
        orig_cols = name_df.columns

        # * Filter case not good accent
        good_name_mask = name_df[name_col].apply(self.__is_good_name)
        good_accent_name_df = name_df[good_name_mask]
        bad_accent_name_df = name_df[~good_name_mask]

        # * Cleansing invalid base element
        bad_accent_name_df[f"clean_{name_col}"] = bad_accent_name_df[
            name_col
        ].apply(remove_invalid_base_element)

        # name_df[f'clean_{name_col}'] = preprocess_df(
        #     name_df,
        #     name_col=name_col,
        #     clean_name=True,
        #     remove_pronoun=True
        # )[name_col]

        # * Separate 1-word strange name
        one_word_mask = (
            bad_accent_name_df[f"clean_{name_col}"].str.split(" ").str.len()
            == 1
        )
        strange_name_mask = (one_word_mask) & (
            ~self.name_process.check_name_valid(
                bad_accent_name_df[one_word_mask][f"clean_{name_col}"]
            )
        )
        one_word_strange_names = bad_accent_name_df[strange_name_mask]
        normal_names = bad_accent_name_df[~strange_name_mask]

        # * Removing nicknames
        predicted_name = remove_nicknames(
            normal_names, name_col=f"clean_{name_col}"
        )

        # predicted_name[['last_name', 'middle_name', 'first_name']] =\
        #     predicted_name[f'clean_{name_col}'].apply(
        #         self.name_process.SplitName).tolist()

        # predicted_name[f'clean_{name_col}'] =\
        #     predicted_name[['last_name', 'middle_name', 'first_name']]\
        #     .fillna('').agg(' '.join, axis=1)\
        #     .str.strip()

        # predicted_name = predicted_name.drop(
        #     columns=['last_name', 'middle_name', 'first_name']
        # )

        # print("Filling diacritics to names...")
        # start_time = time()
        predicted_name["predict"] = predicted_name[f"clean_{name_col}"].apply(
            self.predict_non_accent
        )
        # mean_predict_time = (time() - start_time) / n_names

        # print(f"\nAVG prediction time : {mean_predict_time}s")

        # print('\n\n')

        # print("Applying rule-based postprocess...")
        # start_time = time()
        predicted_name["final"] = predicted_name.apply(
            lambda row: rule_base_name(
                row["predict"], row[f"clean_{name_col}"], self.name_dicts
            ),
            axis=1,
        )

        predicted_name["final"] = predicted_name.apply(
            lambda row: self.choose_better_enrich(
                row[f"clean_{name_col}"], row["final"]
            ),
            axis=1,
        )

        # * Final postprocess
        predicted_name["final"] = replace_trash_string(
            predicted_name, replace_col="final"
        )
        predicted_name["final"] = predicted_name["final"].apply(
            remove_duplicated_name
        )
        predicted_name["final"] = predicted_name["final"].apply(
            remove_invalid_element
        )

        # * Full fill the data
        predicted_name = pd.concat([predicted_name, one_word_strange_names])

        # mean_rb_time = (time() - start_time) / n_names

        # print(f"\nAVG rb time : {mean_rb_time}s")
        out_cols = [
            f"clean_{name_col}",
            "final",
            "predict",
        ]

        for col in out_cols:
            good_accent_name_df[col] = good_accent_name_df[name_col]

        final_name = pd.concat(
            [good_accent_name_df, predicted_name[[*orig_cols, *out_cols]]]
        )

        return final_name

    def unify_name(
        self,
        name_df: pd.DataFrame,
        name_col: str,
        key_col: str,
        keep_cols: List[str],
    ):
        name_df[key_col] = name_df[key_col].astype("str")
        best_name_df = self.name_process.CoreBestName(
            name_df, name_col=name_col, key_col=key_col
        )

        return best_name_df[
            [key_col, name_col, "best_name", "similarity_score", *keep_cols]
        ]
