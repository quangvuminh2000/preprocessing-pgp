"""
Module to extract name from email using rule-based
"""
import math
import re
from typing import Tuple

import pandas as pd
from unidecode import unidecode

from preprocessing_pgp.email.extractors.const import FULLNAME_DICT
from preprocessing_pgp.email.utils import (
    clean_email_name,
)
from preprocessing_pgp.name.type.extractor import process_extract_name_type
from preprocessing_pgp.name.gender.predict_gender import process_predict_gender
from preprocessing_pgp.name.enrich_name import process_enrich

pd.options.mode.chained_assignment = None


class ZipfName:
    """
    Class contains necessary calculation based on Zipfian distribution
    """

    def __init__(self, vocab):
        self.vocab = vocab
        self.vocab_size = len(self.vocab)
        self.__build_cost_dict()

    def __build_cost_dict(self):
        self.word_cost = dict(
            (k, math.log((i+1)*math.log(self.vocab_size)))
            for i, k in enumerate(self.vocab)
        )
        self.max_word_length = max(len(word) for word in self.vocab)

    def extract_component(self, in_str: str) -> str:
        """
        Use dynamic programming to infer the location of spaces in a normed string,
        then return the minimal cost string -- with spaces

        Parameters
        ----------
        in_str : str
            The input normed string

        Returns
        -------
        str
            The returned results of space separated string using Zipfian's law
        """

        cost = [0]

        def get_best_match(str_len: int) -> int:
            """
            Return the best match cost given the len of the string
            """
            start_idx = max(0, str_len-self.max_word_length)
            candidates = enumerate(reversed(
                cost[start_idx:str_len]
            ))

            return min(
                [(c+self.word_cost.get(in_str[str_len-k-1:str_len], 9e999), k+1)
                 for k, c in candidates],
                key=lambda x: x[0]
            )

        # Build the cost array
        in_length = len(in_str)
        for str_len in range(1, in_length+1):
            best_cost, _ = get_best_match(str_len)
            cost.append(best_cost)

        # Backtrack to recover the minimal-cost string
        final_str = []
        str_len = in_length
        while str_len > 0:
            best_cost, k = get_best_match(str_len)
            assert best_cost == cost[str_len], "Wrong logic of calculating cost for string"
            final_str.append(in_str[str_len-k:str_len])
            str_len -= k

        return ' '.join(reversed(final_str))


class EmailNameExtractor:
    """
    Class contains logic to extract username from email
    """

    def __init__(self):
        self.zipf_name = ZipfName(FULLNAME_DICT['name_norm'].tolist())
        self.name_norm_dict =\
            FULLNAME_DICT.set_index('name_norm')\
            .to_dict()['name']

    # ? EXTRACTION COMPONENT OF NAMES
    def _get_username_candidate(
        self,
        data: pd.DataFrame,
        email_name_col: str = 'email_name'
    ) -> pd.DataFrame:
        """
        Extracting username candidate from email's name using ZipF rule

        Parameters
        ----------
        data : pd.DataFrame
            The original dataframe
        email_name_col : str, optional
            The column name contains email_name, by default 'email_name'

        Returns
        -------
        pd.DataFrame
            Component data for username candidate
        """
        data[
            'username_candidate'
        ] = data[
            email_name_col
        ].apply(self.zipf_name.extract_component)

        candidate_data = data['username_candidate'].astype(str)\
            .str.split(' ', expand=True)

        return candidate_data

    def _get_username_from_candidate(
        self,
        data: pd.DataFrame,
        candidate_data: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Combining candidates of name to make a complete name

        Parameters
        ----------
        data : pd.DataFrame
            Original data
        candidate_data : pd.DataFrame
            Candidate data for name

        Returns
        -------
        pd.DataFrame
            Original data with additional column to get the extracted username
            * `username_extracted` : The extracted name from email
        """

        data['username_extracted'] = ''

        for col in candidate_data.columns:
            # * Whether the candidate is good
            good_name_mask =\
                (candidate_data[col].str.len() > 1) &\
                (candidate_data[col].isin(FULLNAME_DICT['name_norm']))

            # * Translate good candidate to their best name
            candidate_data.loc[
                good_name_mask,
                col
            ] = candidate_data.loc[
                good_name_mask,
                col
            ].map(self.name_norm_dict)

            # * Combining to previous candidates
            data.loc[
                good_name_mask,
                'username_extracted'
            ] = data.loc[
                good_name_mask,
                'username_extracted'
            ] + ' ' + candidate_data.loc[
                good_name_mask,
                col
            ]

        # * Cleansing the spare spaces
        data['username_extracted'] =\
            data['username_extracted'].str.strip()

        return data

    def extract_username(
        self,
        data: pd.DataFrame,
        email_name_col: str = 'email_name',
    ) -> pd.DataFrame:
        """
        Generate username from email data

        Parameters
        ----------
        data : pd.DataFrame
            Data containing the email
        email_name_col : str, optional
            Name of email column from data, by default 'email_name'

        Returns
        -------
        pd.DataFrame
            Data with additional columns:
            * `username_extracted` : The username extracted from email
            * `gender_extracted` : The gender predicted from the extracted username
        """

        extracted_data = data.copy()
        orig_cols = extracted_data.columns

        # * Clean the data
        extracted_data = clean_email_name(
            extracted_data,
            email_name_col
        )

        # * Extracting the customer type -- only use 'customer'
        extracted_data = process_extract_name_type(
            extracted_data,
            name_col=f'cleaned_{email_name_col}',
            n_cores=1,
            logging_info=False
        )
        extracted_data[f'cleaned_{email_name_col}'] =\
            extracted_data[f'cleaned_{email_name_col}'].str.lower()

        # * Only proceed clean, 'customer' name
        proceed_mask =\
            (extracted_data[f'cleaned_{email_name_col}'].notna()) &\
            (extracted_data['customer_type'] == 'customer')
        proceed_data = extracted_data[proceed_mask]
        ignored_data = extracted_data[~proceed_mask]

        # * Get email's name candidate
        name_candidate = self._get_username_candidate(
            proceed_data,
            email_name_col=f'cleaned_{email_name_col}'
        )

        # * Extracting username from email
        proceed_data = self._get_username_from_candidate(
            proceed_data,
            name_candidate
        )

        # * Enrich new names
        proceed_data.drop(columns=['customer_type'], inplace=True)
        proceed_data = process_enrich(
            proceed_data,
            name_col='username_extracted',
            n_cores=1
        )

        # * Predict gender from extracted username
        proceed_data = process_predict_gender(
            proceed_data,
            name_col='final',
            n_cores=1,
            logging_info=False
        )
        proceed_data.rename(columns={
            'gender_predict': 'gender_extracted',
            'final': 'enrich_name'
        }, inplace=True)

        # * Combine to get final data
        final_data = pd.concat([
            proceed_data,
            ignored_data
        ])

        return final_data[[
            *orig_cols,
            f'cleaned_{email_name_col}',
            'customer_type',
            'username_extracted',
            'enrich_name',
            'gender_extracted'
        ]]
