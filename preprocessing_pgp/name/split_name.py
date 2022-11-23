import re
from time import time

from unidecode import unidecode
import numpy as np

import pandas as pd
import multiprocessing as mp
from tqdm import tqdm

tqdm.pandas()


def BuildLastName(base_path):
    # load stats lastname
    stats_lastname_vn = pd.read_parquet(
        f'{base_path}/stats_lastname_vn.parquet')

    last_name_list1 = list(
        stats_lastname_vn[stats_lastname_vn['No'] <= 50]['Last_Name'].unique())
    last_name_list2 = ['Nguyễn', 'Trần', 'Lê', 'Phạm', 'Hoàng', 'Huỳnh', 'Phan',
                       'Vũ', 'Võ', 'Đặng', 'Bùi', 'Đỗ', 'Hồ', 'Ngô', 'Dương', 'Lý',
                       'Trương', 'Bùi', 'Đinh', 'Lương', 'Tạ', 'Quách', 'Hứa']
    last_name_list3 = last_name_list1
    last_name_list = list(set(last_name_list3) | set(
        last_name_list1) | set(last_name_list2))

    # return
    return last_name_list1, last_name_list2, last_name_list3, last_name_list


def BuildWordName(base_path):
    # load database name
    ext_data_name1 = pd.read_parquet(f'{base_path}/ext_data.parquet')
    ext_data_name2 = pd.read_parquet(f'{base_path}/ext_data_uit.parquet')

    ext_data_name1.columns = ['full_name', 'gender']
    ext_data_name1['full_name'] = ext_data_name1['full_name'].str.replace(
        '\s+', ' ', regex=True).str.title()
    ext_data_name1['first_name'] = ext_data_name1['full_name'].str.split(
        ' ').str[-1]
    ext_data_name1['last_name_group'] = ext_data_name1['full_name'].str.split(
        ' ').str[0]
    ext_data_name1['last_name'] = ext_data_name1['full_name'].str.split(
        ' ').str[:-1].str.join(' ')

    ext_data_name2['full_name'] = ext_data_name2['full_name'].str.replace(
        '\s+', ' ', regex=True).str.title()
    ext_data_name2 = ext_data_name2[[
        'full_name', 'gender', 'first_name', 'last_name_group', 'last_name']].copy()

    # stats freq
    ext_data_name1['full_name_unicecode'] = ext_data_name1['full_name'].str.lower(
    ).apply(unidecode)
    stats_word_name1 = ext_data_name1['full_name_unicecode'].str.split(
        expand=True).stack().value_counts().reset_index()
    stats_word_name1.columns = ['Word', 'Frequency']

    ext_data_name2['full_name_unicecode'] = ext_data_name2['full_name'].str.lower(
    ).apply(unidecode)
    stats_word_name2 = ext_data_name2['full_name_unicecode'].str.split(
        expand=True).stack().value_counts().reset_index()
    stats_word_name2.columns = ['Word', 'Frequency']

    # wordName
    stats_word_name = pd.concat(
        [stats_word_name1, stats_word_name2], axis=0, ignore_index=False)
    # stats_word_name = stats_word_name1.append(
    # stats_word_name2, ignore_index=False)
    stats_word_name = stats_word_name.groupby(
        by=['Word'])['Frequency'].sum().reset_index()
    stats_word_name = stats_word_name[~((stats_word_name['Frequency'] < 5) |
                                        stats_word_name['Word'].str.contains('[^a-z]'))]
    word_name = set(stats_word_name['Word'])

    # return
    return word_name


class NameProcess:
    def __init__(self, base_path):
        self.word_name = np.array(list(BuildWordName(base_path)))
        self.last_name_list1, self.last_name_list2, self.last_name_list3, self.last_name_list = BuildLastName(
            base_path)
        self.brief_name = {
            'ng': 'nguyen'
        }

    def CountNameVN(self, text):
        try:
            text = unidecode(text.lower())

            # Contains 1 char
            if len(text) == 1:
                return 1

            # Check in word_name
            word_text = np.array(text.split())

            # Handle case where 2 word with same decode in name
            intersect_bool = np.in1d(word_text, self.word_name)
            intersect_score = np.sum(intersect_bool)

            # name containing brief 1 word and brief name
            brief_score = 0
            for word in word_text:
                if len(word) == 1 or word in self.brief_name.keys():
                    brief_score += 1

            return intersect_score + brief_score

        except:
            return -1

    def CleanName(self, raw_name):
        try:
            process_name = raw_name.lower().strip()

            # is email?
            regex_email = r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+'
            process_name = re.sub(regex_email, '', process_name)
            # print(f'email char: {process_name}')

            # is phone?
            regex_phone = r'[0-9]*'
            process_name = re.sub(regex_phone, '', process_name)

            # special char
            process_name = re.sub(r'\(.*\)', '', process_name)
            process_name = re.sub(r'\.|,|-|_', '', process_name)
            process_name = process_name.split(',')[0]
            process_name = process_name.split('-')[0]
            process_name = process_name.strip('-| |.|,|(|)')
            # print(f'special char: {process_name}')

            # fix typing
            regex_typing1 = '0[n|a|i|m|c|t][^wbmc\d ]'
            regex_typing2 = '[h|a|b|v|g|e|x|c]0[^0-9.-]'
            process_name = re.sub(regex_typing1, 'o', process_name)
            process_name = re.sub(regex_typing2, 'o', process_name)
            # print(f'fix typing char: {process_name}')

            # pronoun
            regex_pronoun1 = r'^(?:\bkh\b|\bkhach hang\b|\bchị\b|\bchi\b|\banh\b|\ba\b|\bchij\b|\bc\b|\be\b|\bem\b|\bcô\b|\bco\b|\bchú\b|\bbác\b|\bbac\b|\bme\b|\bdì\b|\bông\b|\bong\b|\bbà\b|\ba\.|\bc\.)\s+'
            regex_pronoun2 = r'^(?:\bnội\b|\bngoại\b)\s+'
            regex_pronoun3 = r'^(?:\bvo anh\b|\bvo a\b|\bvo chu\b|\bbo anh\b|\bme anh\b|\bem anh\b|\bbo a\b|\bban chi\b|\bbo chi\b|\bban\b|\bck\b|\bvk\b)\s+'

            process_name = re.sub(regex_pronoun1, '', process_name)
            process_name = re.sub(regex_pronoun2, '', process_name)
            process_name = re.sub(regex_pronoun3, '', process_name)
            process_name = process_name.strip('-| |.|,|(|)')
            # print(f'remove pronoun char: {process_name}')

            # defaut
            regex_default1 = r'người mua|người nhận|số ngoài danh bạ|nhập số điện thoại|giao hàng|test|[~!#$%^?]'
            regex_default2 = r'dung chung|ky thuat|du phong|dong tien|chu cu|chung phong|co quan|thu nhat|thu hai|so cong ty|chu moi|nhan su|dong nghiep|lien quan|em cua|may ban|so may|nghi lam|quan ly|dat so|su dung|nhan vien|chu nha|moi mua|dien thoai|chuyen di|lap dat|cung phong|nham so|hop dong|tong dai|can ho|ke toan|k co|so phu|lien he|lien lac|don di|so cu|so moi|ve que|khong dung|ben canh|ko co'
            regex_default3 = r'kh |khach hang|chu nha|cong ty|cc|nsd|vo kh|chong kh|so chu hd|so moi|nguoi moi tiep nhan'
            process_name = re.sub(regex_default1, '', process_name)
            process_name = re.sub(regex_default2, '', process_name)
            process_name = re.sub(regex_default3, '', process_name)

            process_name = process_name.strip('-| |.|,|(|)')
            process_name = re.sub(r'\s+', ' ', process_name)
            # print(f'remove noun char: {process_name}')

            # is dict name VN
            pct_vn = self.CountNameVN(process_name) / \
                len(process_name.split(' '))
#             print(pct_vn)
            process_name = None if ((pct_vn < 0.8) |
                                    # (len(process_name) == 1) |
                                    (len(process_name.split(' ')) > 6)
                                    ) else process_name
            # print(f'check VN char: {process_name}')

            # title
            process_name = process_name.title()

            return process_name

        except:
            return None

    # SPLIT_NAME

    def SplitName(self, full_name):
        try:
            # Variable
            use_only_last_name = True
            full_name = full_name.replace('\s+', ' ').strip().title()
            last_name = ''
            middle_name = None
            first_name = None

            # Case 0: B
            if (full_name.split(' ') == 1):
                # print('Case 0')
                first_name = full_name
                return last_name, middle_name, first_name

            # Case 1: Nguyen Van C
            check_end_case1 = False
            while (check_end_case1 == False):
                for key_vi in self.last_name_list1:
                    key_vi = key_vi + ' '

                    is_case11 = (full_name.find(key_vi) == 0)
                    is_case12 = (full_name.find(unidecode(key_vi)) == 0)
                    is_case1 = is_case11 or is_case12
                    if is_case1:
                        # print('Case 1')
                        key = key_vi if is_case11 else unidecode(key_vi)

                        last_name = (last_name + ' ' + key).strip()
                        full_name = full_name.replace(key, '', 1).strip()

                        if (use_only_last_name == True):
                            check_end_case1 = True
                        break

                    if (full_name.split(' ') == 1) or (key_vi.strip() == self.last_name_list1[-1]):
                        check_end_case1 = True

            # Case 2: Van D Nguyen
            if (last_name.strip() == ''):
                check_end_case2 = False
                while (check_end_case2 == False):
                    for key_vi in self.last_name_list2:
                        key_vi = ' ' + key_vi

                        is_case21 = (len(full_name)-full_name.rfind(key_vi)
                                     == len(key_vi)) & (full_name.rfind(key_vi) != -1)
                        is_case22 = (len(full_name)-full_name.rfind(unidecode(key_vi)) == len(
                            unidecode(key_vi))) & (full_name.rfind(unidecode(key_vi)) != -1)

                        is_case2 = is_case21 or is_case22
                        if is_case2:
                            # print('Case 2')
                            key = key_vi if is_case21 else unidecode(key_vi)

                            last_name = (key + ' ' + last_name).strip()
                            full_name = ''.join(
                                full_name.rsplit(key, 1)).strip()

                            if (use_only_last_name == True):
                                check_end_case2 = True
                            break

                        if (full_name.split(' ') == 1) or (key_vi.strip() == self.last_name_list2[-1]):
                            check_end_case2 = True

            # Case 3: E Nguyen Van
            if (last_name.strip() == ''):
                temp_full_name = full_name
                temp_first_name = temp_full_name.split(' ')[0]
                temp_full_name = ' '.join(
                    temp_full_name.split(' ')[1:]).strip()

                check_end_case3 = False
                while (check_end_case3 == False):
                    for key_vi in self.last_name_list3:
                        key_vi = key_vi + ' '

                        is_case31 = (temp_full_name.find(key_vi) == 0)
                        is_case32 = (temp_full_name.find(
                            unidecode(key_vi)) == 0)
                        is_case3 = is_case31 or is_case32
                        if is_case3:
                            # print('Case 3')
                            key = key_vi if is_case31 else unidecode(key_vi)

                            last_name = (last_name + ' ' + key).strip()
                            temp_full_name = temp_full_name.replace(
                                key, '', 1).strip()

                            if (use_only_last_name == True):
                                check_end_case3 = True
                            break

                        if (full_name.split(' ') == 1) or (key_vi.strip() == self.last_name_list3[-1]):
                            check_end_case3 = True

                if (last_name.strip() != ''):
                    first_name = temp_first_name
                    middle_name = temp_full_name

                    return last_name, middle_name, first_name

            # Fillna
            first_name = full_name.split(' ')[-1]
            try:
                full_name = ''.join(full_name.rsplit(first_name, 1)).strip()
                middle_name = full_name
            except:
                # print('Case no middle name')
                middle_name = None

            # Replace '' to None
            last_name = None if (last_name == '') else last_name
            middle_name = None if (middle_name == '') else middle_name
            first_name = None if (first_name == '') else first_name

            return last_name, middle_name, first_name

        except:
            return None, None, None

    def unidecode_with_na(self, name):
        if name is None:
            return 'NONE'
        return unidecode(name)
    
    def CoreBestName(self, raw_names_n, name_col='name', key_col='phone'):
        start_time = time()
        raw_names_n = raw_names_n[raw_names_n[name_col].notna()]
        # Rename name column
        raw_names_n = raw_names_n.rename(columns={name_col: 'raw_name'})
        # Skip name (non personal)
        map_name_customer = raw_names_n[['raw_name']].copy().drop_duplicates()
        map_name_customer.columns = ['name']
        map_name_customer['num_word'] = map_name_customer['name'].str.split(
            ' ').str.len()
        skip_names = map_name_customer[(
            map_name_customer['num_word'] > 5)]['name'].unique()
        skip_names_df = raw_names_n[raw_names_n['raw_name'].isin(
            skip_names)][[key_col, 'raw_name']].copy().drop_duplicates()
        names_df = raw_names_n[~raw_names_n['raw_name'].isin(
            skip_names)][[key_col, 'raw_name']].copy().drop_duplicates()
        print(">> Skip/Filter name")
        # Split name: last, middle, first
        map_split_name = names_df[['raw_name']].copy().drop_duplicates()
        with mp.Pool(8) as pool:
            map_split_name[['last_name', 'middle_name', 'first_name']
                           ] = pool.map(self.SplitName, map_split_name['raw_name'])
        names_df = names_df.merge(map_split_name, how='left', on=['raw_name'])
        # Create group_id -> by firstname
        print(names_df[names_df['first_name'].isna()])
        names_df['unidecode_first_name'] = names_df['first_name'].progress_apply(
            self.unidecode_with_na)
        names_df['group_id'] = names_df[key_col] + \
            '-' + names_df['unidecode_first_name']
        names_df = names_df.drop(columns=['unidecode_first_name'])
        # Split case process best_name
        names_df.loc[names_df['last_name'].notna(),
                     'unidecode_last_name'] = names_df.loc[names_df['last_name'].notna(), 'last_name'].apply(unidecode)
        names_df['num_last_name'] = names_df.groupby(
            by=['group_id'])['unidecode_last_name'].transform('nunique')
        names_df['mode_last_name'] = names_df.groupby(by=['group_id'])['unidecode_last_name'].progress_transform(lambda x: x.mode())
#         names_df['type'] = names_df.groupby(by=['group_id'])['mode_last_name'].transform(lambda x: x.apply(lambda y:  if y==None))
        info_name_columns = ['group_id', 'raw_name',
                             'last_name', 'middle_name', 'first_name']
        # In TESTING PROGRESS
        n_lastname_mask = (names_df['num_last_name'] >= 2) \
                              & (names_df['mode_last_name'] != names_df['unidecode_last_name']) \
                              & (names_df['unidecode_last_name'] != None)
        names_n_df = names_df[n_lastname_mask][info_name_columns].copy()
        names_1_df = names_df[~n_lastname_mask][info_name_columns].copy()
#         return names_n_df, names_1_df
        print(">> Create group_id")
        # Process case: 1 first_name - n last_name
        post_names_n_df = names_n_df[names_n_df['last_name'].isna()].copy()
        map_names_n_df = names_n_df[names_n_df['last_name'].notna() &
                                    names_n_df['group_id'].isin(post_names_n_df['group_id'])].copy()
        map_names_n_df['num_char'] = map_names_n_df['raw_name'].str.len()
        map_names_n_df['num_word'] = map_names_n_df['raw_name'].str.split(
            ' ').str.len()
        map_names_n_df['accented'] = map_names_n_df['raw_name'] != map_names_n_df['raw_name'].apply(
            unidecode)
        map_names_n_df = map_names_n_df.sort_values(
            by=['group_id', 'num_word', 'num_char', 'accented'], ascending=False)
        map_names_n_df = map_names_n_df.groupby(by=['group_id']).head(1)
        map_names_n_df = map_names_n_df[['group_id', 'raw_name']].rename(
            columns={'raw_name': 'best_name'})
        post_names_n_df = post_names_n_df.merge(
            map_names_n_df, how='left', on=['group_id'])
        post_names_n_df = post_names_n_df[[
            'group_id', 'raw_name', 'best_name']]
        names_n_df = names_n_df.merge(post_names_n_df, how='left', on=[
                                      'group_id', 'raw_name'])
        names_n_df.loc[names_n_df['best_name'].isna(
        ), 'best_name'] = names_n_df['raw_name']
        names_n_df = names_n_df[['group_id', 'raw_name', 'best_name']]
        print(">> 1 first_name - n last_name")
        # Process case: 1 first_name - 1 last_name
        map_names_1_df = names_1_df[['group_id']].drop_duplicates()
        for element_name in tqdm(['last_name', 'middle_name', 'first_name']):
            # filter data detail
            map_element_name = names_1_df[names_1_df[element_name].notna(
            )][['group_id', element_name]].copy().drop_duplicates()
            # create features
            map_element_name[f'unidecode_{element_name}'] = map_element_name[element_name].apply(
                unidecode)
            map_element_name['num_overall'] = map_element_name.groupby(
                by=['group_id', f'unidecode_{element_name}'])[element_name].transform('count')
            map_element_name = map_element_name.drop(
                columns=f'unidecode_{element_name}')
            map_element_name['num_char'] = map_element_name[element_name].str.len()
            map_element_name['num_word'] = map_element_name[element_name].str.split(
                ' ').str.len()
            map_element_name['accented'] = map_element_name[element_name] != map_element_name[element_name].apply(
                unidecode)
            # approach to choice best
            # map_element_name = map_element_name.sort_values(by=['group_id', 'num_overall', 'num_char', 'num_word', 'accented'], ascending=False)
#             map_element_name = map_element_name.sort_values(
#                 by=['group_id', 'num_word', 'accented', 'num_char', 'num_overall'], ascending=False)
            if element_name != 'middle_name':
                map_element_name = map_element_name.sort_values(
                    by=['group_id',
                        'accented',
                        'num_char',
                        'num_word',
                        'num_overall'], ascending=False)
            else: # For middlename we choose the longest name then accent latter
                map_element_name = map_element_name.sort_values(
                    by=['group_id',
                        'num_word',
                        'accented',
                        'num_char',
                        'num_overall'], ascending=False)
            map_element_name = map_element_name.groupby(
                by=['group_id']).head(1)
            map_element_name = map_element_name[['group_id', element_name]]
            map_element_name.columns = ['group_id', f'best_{element_name}']
            # merge
            map_names_1_df = map_names_1_df.merge(
                map_element_name, how='left', on=['group_id'])
            map_names_1_df.loc[map_names_1_df[f'best_{element_name}'].isna(
            ), f'best_{element_name}'] = None
        # combine element name
#         return map_names_1_df
        dict_trash = {'': None, 'Nan': None, 'nan': None, 'None': None,
                      'none': None, 'Null': None, 'null': None, "''": None}
        columns = ['best_last_name', 'best_middle_name', 'best_first_name']
        ### BIG BUG WHEN NO ENTRY
        map_names_1_df['best_name'] = map_names_1_df[columns].fillna('').agg(' '.join, axis=1).str.replace(
            '(?<![a-zA-Z0-9]),', '', regex=True).str.replace('-(?![a-zA-Z0-9])', '', regex=True)
        ### END BUG
        map_names_1_df['best_name'] = map_names_1_df['best_name'].str.strip().replace(
            dict_trash)
        map_names_1_df.loc[map_names_1_df['best_name'].isna(),
                           'best_name'] = None
        # merge
        names_1_df = names_1_df.merge(
            map_names_1_df[['group_id', 'best_name']], how='left', on=['group_id'])
        names_1_df = names_1_df[['group_id', 'raw_name', 'best_name']]
        print(">> 1 first_name - 1 last_name")
        # Concat
        names_df = pd.concat([names_1_df, names_n_df], ignore_index=True)
        # Calculate similarity_score
        name_list_1 = list(names_df['raw_name'].unique())
        name_list_2 = list(names_df['best_name'].unique())
        map_element_name = pd.DataFrame()
        map_element_name['name'] = list(set(name_list_1) | set(name_list_2))
        with mp.Pool(8) as pool:
            map_element_name[['last_name', 'middle_name', 'first_name']] = pool.map(
                self.SplitName, map_element_name['name'])
        for flag in ['raw', 'best']:
            temp = map_element_name.copy()
            temp.columns = [f'{flag}_name', f'{flag}_last_name',
                            f'{flag}_middle_name', f'{flag}_first_name']
            names_df = names_df.merge(temp, how='left', on=[f'{flag}_name'])
        # similar score by element
        for element_name in tqdm(['last_name', 'middle_name', 'first_name']):
            # split data to compare
            condition_compare = names_df[f'raw_{element_name}'].notna(
            ) & names_df[f'best_{element_name}'].notna()
            compare_names_df = names_df[condition_compare].copy()
            not_compare_names_df = names_df[~condition_compare].copy()
            # compare raw with best
            compare_names_df[f'similar_{element_name}'] = compare_names_df[f'raw_{element_name}'].apply(
                unidecode) == compare_names_df[f'best_{element_name}'].apply(unidecode)
            compare_names_df[f'similar_{element_name}'] = compare_names_df[f'similar_{element_name}'].astype(
                int)
            not_compare_names_df[f'similar_{element_name}'] = 1
            # concat
            names_df = pd.concat(
                [compare_names_df, not_compare_names_df], ignore_index=True)
        weights = [0.25, 0.25, 0.5]
        names_df['similarity_score'] = weights[0]*names_df['similar_last_name'] + weights[1] * \
            names_df['similar_middle_name'] + \
            weights[2]*names_df['similar_first_name']
        print(">> similarity_score")
        # Postprocess
        pre_names_df = names_df[['group_id', 'raw_name',
                                 'best_name', 'similarity_score']].copy()
        pre_names_df[key_col] = pre_names_df['group_id'].str.split('-').str[0]
        pre_names_df = pre_names_df.drop(columns=['group_id'])
        pre_names_df = pd.concat(
            [pre_names_df, skip_names_df], ignore_index=True)
        pre_names_df.loc[pre_names_df['best_name'].isna(
        ), 'best_name'] = pre_names_df['raw_name']
        pre_names_df.loc[pre_names_df['similarity_score'].isna(),
                         'similarity_score'] = 1
        print(">> Postprocess")
        # Merge
        pre_names_n = raw_names_n.merge(
            pre_names_df, how='left', on=[key_col, 'raw_name'])
        pre_names_n.loc[pre_names_n['best_name'].isna(
        ), 'best_name'] = pre_names_n['raw_name']
        pre_names_n.loc[pre_names_n['similarity_score'].isna(),
                        'similarity_score'] = 1
        # Return
        pre_names_n = pre_names_n.rename(columns={'raw_name': name_col})
        
        unify_time = time()-start_time
        print(f"Unify runs in {unify_time/60} mins")
        
        return pre_names_n


if __name__ == '__main__':
    name = input("Please input the name: ")

    name_process = NameProcess()
    name = name_process.CleanName(name)
    last, middle, first = name_process.SplitName(name)
