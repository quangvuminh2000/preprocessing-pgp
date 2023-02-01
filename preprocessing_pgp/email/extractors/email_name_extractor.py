"""
Module to extract name from email using rule-based
"""
import math
import re
from string import digits

import pandas as pd
from unidecode import unidecode

from preprocessing_pgp.email.extractors.const import FULLNAME_DICT
from preprocessing_pgp.email.utils import clean_email_name


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
        self.__generate_lname_unique()
        self.__generate_mname_unique()
        self.__generate_fname_unique()
        self.__generate_name_dict()
        self.zipf_name = ZipfName(self.name_norm_unique)

    def __normalize_name(self, name):
        # Unidecode, lower and remove spacing
        de_name = unidecode(name)
        de_name = de_name.lower()
        norm_name = de_name.replace(' ', '')
        return norm_name

    def __generate_lname_unique(self):
        if hasattr(self, 'lname_unique'):
            return

        lname_series = FULLNAME_DICT['last_name']
        lname_series = lname_series.reindex(
            lname_series.str.len().sort_values(ascending=False).index
        )
        self.lname_unique = lname_series.unique().tolist()
        self.lname_norm_unique = lname_series\
            .apply(self.__normalize_name)\
            .unique().tolist()
        self.lname_norm_unique_regex =\
            '|'.join(
                self.lname_norm_unique
            )

    def __generate_fname_unique(self):
        if hasattr(self, 'fname_unique'):
            return

        fname_series = FULLNAME_DICT['first_name']
        fname_series = fname_series.reindex(
            fname_series.str.len().sort_values(ascending=False).index
        )
        self.fname_unique = fname_series.unique().tolist()
        self.fname_norm_unique = fname_series\
            .apply(self.__normalize_name)\
            .unique().tolist()
        self.fname_norm_unique_regex =\
            '|'.join(
                self.fname_norm_unique
            )

    def __generate_mname_unique(self):
        if hasattr(self, 'mname_unique'):
            return

        mname_series = FULLNAME_DICT[FULLNAME_DICT['middle_name']
                                     != '']['middle_name']
        mname_series = mname_series.reindex(
            mname_series.str.len().sort_values(ascending=False).index
        )
        self.mname_unique = mname_series.unique().tolist()
        self.mname_norm_unique = mname_series\
            .apply(self.__normalize_name)\
            .unique().tolist()
        self.mname_norm_unique_regex =\
            '|'.join(
                self.mname_norm_unique
            )
        self.mname_norm_trace_dict = dict(zip(
            self.mname_unique,
            self.mname_norm_unique
        ))

    def __generate_name_dict(self):
        if hasattr(self, 'name_unique'):
            return

        # * Create dataframe for names
        names_df = pd.DataFrame({
            'name': [
                *self.lname_unique,
                *self.mname_unique,
                *self.fname_unique
            ]
        })
        names_df['length'] = names_df['name'].str.len()
        names_df = names_df.sort_values(by='length', ascending=False)
        names_df['name_norm'] = names_df['name'].apply(self.__normalize_name)

        self.name_norm_unique = names_df['name_norm'].unique().tolist()
        self.name_norm_trace_dict = pd.Series(
            names_df['name'].values,
            index=names_df['name_norm']
        ).to_dict()

    def extract_fullname(
        self,
        email_name: str
    ) -> str:
        if email_name == '':
            return ''

        # * Extract firstname
        cleaned_email_name = email_name.translate(str.maketrans('', '', digits))
        found_fname = re.findall(self.fname_norm_unique_regex, cleaned_email_name)
        if len(found_fname) == 0:
            return ''
        final_fname = found_fname[0]
        remain_name = re.sub(final_fname, '', cleaned_email_name)

        # * Extract lastname
        found_lname = re.findall(self.lname_norm_unique_regex, remain_name)
        final_lname = ''
        if len(found_lname) != 0:
            final_lname = found_lname[0]
            remain_name = re.sub(final_lname, '', remain_name)

        # * Extract middlename
        found_mname = re.findall(self.mname_norm_unique_regex, remain_name)
        final_mname = ''
        if len(found_mname) != 0:
            final_mname = found_mname[0]
        else:
            final_mname = remain_name

        # if len(found_lname) != 0:
        #     final_lname = found_lname[0]
        #     final_fname = found_fname[-1]
        #     found_mname = re.findall(self.mname_norm_unique_regex, remain_name)
        #     if len(found_mname) == 0:
        #         final_mname = remain_name
        #     else:
        #         final_mname = found_mname[0]
        # else:
        #     if len(found_fname) > 1:
        #         last_names = [name for name in found_fname
        #                        if name in self.lname_norm_unique]
        #         if len(last_names) == 0:
        #             final_lname = ''
        #             final_fname = found_fname[-1]
        #         else:
        #             final_lname = last_names[0]
        #             found_fname.remove(final_lname)
        #             final_fname = found_fname[-1]
        #         found_mname = re.findall(self.mname_norm_unique_regex, remain_name)
        #         if len(found_mname) == 0:
        #             final_mname = remain_name
        #         else:
        #             final_mname = found_mname[0]

        full_name = ' '.join([final_lname, final_mname, final_fname])
        return re.sub(' +', ' ', full_name).strip()

    def extract_username(
        self,
        data: pd.DataFrame,
        email_name_col: str = 'email_name',
    ) -> pd.DataFrame:

        extracted_data = data.copy()
        extracted_data['username_extracted'] =\
            extracted_data[email_name_col].apply(self.extract_fullname)

        return extracted_data
        # orig_cols = extracted_data.columns

        # # * Clean email's name
        # extracted_data[f'cleaned_{email_name_col}'] =\
        #     extracted_data[email_name_col].apply(clean_email_name)

        # # * Extract candidate component in names
        # extracted_data['username_candidate'] =\
        #     extracted_data[email_name_col].apply(
        #         self.zipf_name.extract_component)

        # # * Do username logic extraction
        # test = extracted_data['username_candidate']\
        #     .astype(str)\
        #     .str.split(' ', expand=True)

        # extracted_data['username'] = ''
        # extracted_data['last_name_fill'] = False
        # extracted_data['first_name_found'] = None

        # for col in test.columns:
        #     username_valid = test[col].str.len() > 1
        #     lastname_condition = ((test[col].isin(self.lname_norm_unique)))
        #     # Find lastname for the first time
        #     lastname_case_1 = username_valid & lastname_condition & (
        #         (extracted_data['username'].str.len() == 0) | (~extracted_data['last_name_fill']))
        #     # Find another last name
        #     lastname_case_2 = username_valid & lastname_condition & (
        #         extracted_data['username'].str.len() != 0) & (extracted_data['last_name_fill'])

        #     firstname_condition = (test[col].isin(
        #         self.fname_norm_unique)) & (~lastname_condition)
        #     # Find firstname before lastname found
        #     firstname_case_1 = username_valid & firstname_condition & (
        #         ~extracted_data['last_name_fill'])
        #     # Find firstname after lastname
        #     firstname_case_2 = username_valid & firstname_condition & (
        #         extracted_data['last_name_fill'])
        #     test.loc[username_valid, col] = test.loc[username_valid, col].map(
        #         self.name_norm_trace_dict)
        #     extracted_data.loc[lastname_case_1, 'username'] = test.loc[lastname_case_1,
        #                                                                col] + extracted_data.loc[lastname_case_1, 'username']
        #     extracted_data.loc[lastname_case_2,
        #                        'username'] += ' ' + test.loc[lastname_case_2, col]
        #     extracted_data.loc[lastname_case_1, 'last_name_fill'] = True
        #     extracted_data.loc[firstname_case_1 & (extracted_data['first_name_found'].notna(
        #     )), 'first_name_found'] += ' ' + test.loc[firstname_case_1 & (extracted_data['first_name_found'].notna()), col]

        #     extracted_data.loc[firstname_case_1 & (extracted_data['first_name_found'].isna(
        #     )), 'first_name_found'] = test.loc[firstname_case_1 & (extracted_data['first_name_found'].isna()), col]
        #     extracted_data.loc[firstname_case_2, 'username'] = extracted_data.loc[firstname_case_2,
        #                                                                           'username'] + ' ' + test.loc[firstname_case_2, col]
        # extracted_data.loc[extracted_data['first_name_found'].notna(), 'username'] += ' ' + \
        #     extracted_data.loc[extracted_data['first_name_found'].notna(
        #     ), 'first_name_found']
        # extracted_data['username'] = extracted_data['username'].astype(
        #     str).str.strip()
        # extracted_data.loc[extracted_data['username'].astype(
        #     str).str.len() <= 1, 'username'] = None

        # return extracted_data[[*orig_cols, 'username_candidate', 'username']]
