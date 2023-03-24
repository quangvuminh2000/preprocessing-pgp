
import pandas as pd
from unidecode import unidecode
from datetime import datetime, timedelta
from difflib import SequenceMatcher
import sys

import subprocess

sys.path.append('/bigdata/fdp/cdp/source/core_profile/preprocess/utils')
from preprocess_profile import (
    remove_same_username_email,
    extracting_pronoun_from_name
)
from enhance_profile import enhance_common_profile
from filter_profile import get_difference_data
from const import (
    hdfs,
    CENTRALIZE_PATH,
    PREPROCESS_PATH
)

# function get profile change/new


# def DifferenceProfile(now_df, yesterday_df):
#     difference_df = now_df[~now_df.apply(tuple, 1).isin(
#         yesterday_df.apply(tuple, 1))].copy()
#     return difference_df

# function unify profile


def UnifyFtel(
    profile_internet: pd.DataFrame,
    n_cores: int = 1
):
    dict_trash = {
        '': None, 'Nan': None, 'nan': None,
        'None': None, 'none': None, 'Null': None,
        'null': None, "''": None
    }
    dict_location = pd.read_parquet('/data/fpt/ftel/cads/dep_solution/user/namdp11/scross_fill/runner/refactor/material/dict_location.parquet',
                                    filesystem=hdfs)

    # * Processing info
    print(">>> Processing Info")
    profile_internet = profile_internet.rename(columns={'uid': 'contract'})

    # * Enhancing common profile
    # profile_internet = enhance_common_profile(
    #     profile_internet,
    #     n_cores=n_cores
    # )

    # load datapay => customer type
    # ds_contract = pd.read_parquet('/data/fpt/ftel/isc/dwh/ds_contract.parquet',
    #                               columns=['contract', 'net_customer_type'],
    #                               filesystem=hdfs).drop_duplicates(subset=['contract'], keep='last')
    # ds_contract.columns = ['contract', 'datapay_customer_type']
    # profile_ftel = profile_ftel.merge(ds_contract, how='left', on='contract')
    # profile_ftel.loc[profile_ftel['source'] ==
    #                  'multi', 'datapay_customer_type'] = None
    # profile_ftel.loc[profile_ftel['source'] == 'multi', 'city'] = None

    # customer type
    # profile_ftel.loc[profile_ftel['customer_type'].isna(
    # ), 'customer_type'] = profile_ftel['datapay_customer_type']
    # profile_ftel = profile_ftel.drop(columns=['datapay_customer_type'])

    # birthday
    print(">>> Processing Birthday")
    condition_birthday = profile_internet['birthday'].notna()
    profile_internet.loc[condition_birthday, 'birthday'] =\
        pd.to_datetime(
            profile_internet[condition_birthday]['birthday'].astype(str),
            errors='coerce'
    ).dt.strftime('%d/%m/%Y')

    # drop name is username_email
    # print(">>> Extra Cleansing Name")

    # profile_internet = remove_same_username_email(
    #     profile_internet,
    #     name_col='name',
    #     email_col='email'
    # )
    # profile_internet = profile_internet.rename(columns={
    #     'gender_enrich': 'gender'
    # })

    # clean name
    # condition_name = (profile_internet['customer_type'] == 'Ca nhan')\
    #     & (profile_internet['name'].notna())
    # profile_internet = extracting_pronoun_from_name(
    #     profile_internet,
    #     condition=condition_name,
    #     name_col='name',
    # )

    # is full name
    # print(">>> Checking Full Name")
    # profile_internet.loc[profile_internet['last_name'].notna(
    # ) & profile_internet['first_name'].notna(), 'is_full_name'] = True
    # profile_internet['is_full_name'] = profile_internet['is_full_name'].fillna(False)
    # profile_internet = profile_internet.drop(
    #     columns=['last_name', 'middle_name', 'first_name'])

    # unify location
    print(">>> Processing Address")
    norm_ftel_city = pd.read_parquet('/data/fpt/ftel/cads/dep_solution/user/namdp11/scross_fill/runner/refactor/material/ftel_provinces.parquet',
                                     filesystem=hdfs)
    norm_ftel_city.columns = ['city', 'norm_city']

    norm_ftel_district = pd.read_parquet('/data/fpt/ftel/cads/dep_solution/user/namdp11/scross_fill/runner/refactor/material/ftel_districts.parquet',
                                         filesystem=hdfs)
    norm_ftel_district.columns = [
        'district', 'norm_city', 'norm_district', 'new_norm_district']

    # update miss district
    district_update = list(
        set(profile_internet['district']) - set(norm_ftel_district['district']))
    location_dict = pd.read_parquet('/data/fpt/ftel/cads/dep_solution/user/namdp11/scross_fill/runner/refactor/material/location_dict.parquet',
                                    filesystem=hdfs)
    location_dict.columns = ['city', 'district', 'norm_city', 'norm_district']
    district_list = list(location_dict['norm_district'].unique())

    def fix_miss_district(district_list, district):
        df = pd.DataFrame()
        df['district'] = district_list
        df['similar'] = df['district'].apply(
            lambda x: SequenceMatcher(None, district.lower(), x.lower()).ratio())

        df = df[df.similar == df.similar.max()]
        result = df['district'].values[0]
        return result

    miss_district = {}
    for district in district_update:
        if district != None:
            result = fix_miss_district(district_list, district)
            miss_district[district] = result

    # merge city, district
    profile_internet = pd.merge(
        profile_internet.set_index('city'),
        norm_ftel_city.set_index('city'),
        left_index=True, right_index=True,
        how='left',
        sort=False
    )
    profile_internet = profile_internet.merge(
        norm_ftel_district[['district', 'norm_city', 'new_norm_district']],
        how='left', on=['district', 'norm_city']
    )

    # fix bugs location
    profile_internet.loc[profile_internet['contract'].notna() &
                     profile_internet['phone'].notna(), 'contract_phone_ftel'] = profile_internet['phone'] + '-' + profile_internet['contract']
    profile_internet.loc[profile_internet['contract'].notna() &
                     profile_internet['phone'].isna(), 'contract_phone_ftel'] = profile_internet['contract']

    bug_location_ftel = profile_internet[profile_internet['norm_city'].isna(
    ) | profile_internet['new_norm_district'].isna()]
    bug_location_ftel1 = bug_location_ftel[bug_location_ftel['norm_city'].notna(
    )]
    bug_location_ftel2 = bug_location_ftel[~bug_location_ftel['contract_phone_ftel'].isin(
        bug_location_ftel1['contract_phone_ftel'])]

    bug_location_ftel1 = bug_location_ftel1.drop(columns=['new_norm_district'])
    bug_location_ftel1 = bug_location_ftel1.merge(
        norm_ftel_district[['district', 'new_norm_district']],
        how='left', on=['district']
    )

    bug_location_ftel2 = bug_location_ftel2.drop(
        columns=['norm_city', 'new_norm_district'])
    bug_location_ftel2 = bug_location_ftel2.merge(
        norm_ftel_district[['district', 'norm_city', 'new_norm_district']],
        how='left', on=['district']
    )

    bug_location_ftel = bug_location_ftel1.append(
        bug_location_ftel2, ignore_index=True)

    profile_internet = profile_internet[~profile_internet['contract_phone_ftel'].isin(
        bug_location_ftel['contract_phone_ftel'])]
    profile_internet = profile_internet.append(bug_location_ftel, ignore_index=True)

    profile_internet.loc[profile_internet['district'].isin(miss_district.keys(
    )), 'new_norm_district'] = profile_internet['district'].map(miss_district)

    profile_internet['city'] = profile_internet['norm_city']
    profile_internet['district'] = profile_internet['new_norm_district']
    profile_internet = profile_internet.drop(
        columns=['norm_city', 'new_norm_district'])

    # fix distric-city
    dict_district = dict_location[[
        'district', 'city']].drop_duplicates().copy()
    stats_district = dict_district.groupby(
        by=['district'])['city'].agg(num_city='count').reset_index()
    stats_district = stats_district[stats_district['num_city'] == 1]
    dict_district = dict_district[dict_district['district'].isin(
        stats_district['district'])]
    dict_district = dict_district.rename(columns={'city': 'new_city'})
    dict_district = dict_district.append([{'district': 'Quan Thu Duc', 'new_city': 'Thanh pho Ho Chi Minh'},
                                          {'district': 'Quan 9',
                                              'new_city': 'Thanh pho Ho Chi Minh'},
                                          {'district': 'Quan 2', 'new_city': 'Thanh pho Ho Chi Minh'}],
                                         ignore_index=True)

    profile_internet = profile_internet.merge(dict_district, how='left', on='district')
    profile_internet.loc[profile_internet['new_city'].notna(
    ), 'city'] = profile_internet['new_city']
    profile_internet = profile_internet.drop(columns=['new_city'])

    # unify ward
    def UnifyWardFTel1(dict_location, ward, district, city):
        if ward == None:
            return None

        ward = unidecode(ward)
        location = dict_location[['ward', 'district',
                                  'city']].drop_duplicates().copy()

        if city != None:
            location = location[location['city'] == city]
        if district != None:
            if district in ['Quan Thu Duc', 'Quan 9', 'Quan 2']:
                district = 'Thanh pho Thu Duc'
            location = location[location['district'] == district]

        location['ward_temp'] = location['ward'].str.replace(
            'Phuong 0', 'Phuong ')

        location['similar'] = location['ward_temp'].apply(
            lambda x: SequenceMatcher(None, ward.lower(), x.lower()).ratio())

        location = location[(location['similar'] == location['similar'].max()) &
                            (location['similar'] >= 0.8)]

        unify_ward = None
        if location.empty == False:
            unify_ward = location['ward'].values[0]

        return unify_ward

    def UnifyWardFTel2(dict_location, ward, district):
        if ward == None:
            return None

        ward = unidecode(ward)
        location = dict_location[['ward', 'district']].drop_duplicates().copy()

        ward = ward.title().replace('P.', 'Phuong ').replace('F.', 'Phuong ')
        ward = ward.title().replace('T.Tran', 'Thi Tran ').replace(
            'T.T', 'Thi Tran ').replace('Tt', 'Thi Tran ')
        ward = ward.title().replace('Tx', 'Thi Xa ')

        if district != None:
            location = location[location['district'] == district]

        location['ward_temp'] = location['ward'].str.replace(
            'Phuong 0', 'Phuong ')

        location['similar'] = location['ward_temp'].apply(
            lambda x: SequenceMatcher(None, ward.lower(), x.lower()).ratio())

        location = location[(location['similar'] == location['similar'].max()) &
                            (location['similar'] >= 0.8)]

        unify_ward = None
        if location.empty == False:
            unify_ward = location['ward'].values[0]

        return unify_ward

    def UnifyWardFTel3(dict_location, ward, district):
        if ward == None:
            return None

        ward = unidecode(ward)
        location = dict_location[['ward', 'district']].drop_duplicates().copy()

        if district != None:
            location = location[location['district'] == district]

        ward = ward.title().replace('P.', '').replace('F.', '')
        ward = ward.title().replace('T.Tran', '').replace('T.T', '').replace('Tt', '')
        ward = ward.title().replace('Tx', '')

        location['ward_temp'] = location['ward'].str.replace(
            'Phuong 0', 'Phuong ')
        location['ward_temp'] = location['ward_temp'].replace({'Phuong ': '',
                                                               'Xa ': '',
                                                               'Thi Tran ': '',
                                                               'Thi Xa ': ''}, regex=True)

        location['similar'] = location['ward_temp'].apply(
            lambda x: SequenceMatcher(None, ward.lower(), x.lower()).ratio())

        location = location[(location['similar'] == location['similar'].max()) &
                            (location['similar'] >= 0.8)]

        unify_ward = None
        if location.empty == False:
            unify_ward = location['ward'].values[0]

        return unify_ward

    def UnifyWardFTel4(ward):
        if ward == None:
            return None

        ward = unidecode(ward)
        ward = ward.title()
        unify_ward = None
        for key in ['Xa', 'Phuong', 'Huyen', 'Thi Tran', 'T.Tran', 'Tt', 'T.T', 'P.', 'F.', 'Thi Xa', 'Tx']:
            check_idx = ward.find(key)
            if check_idx == 0:
                if key in ['Xa', 'Phuong', 'Huyen']:
                    unify_ward = ward
                elif key in ['Thi Tran', 'T.Tran', 'Tt', 'T.T']:
                    unify_ward = ward.replace(key, 'Thi tran')
                elif key in ['P.', 'F.']:
                    unify_ward = ward.replace(key, 'Phuong ')
                elif key in ['Thi Xa', 'Tx']:
                    unify_ward = ward.replace(key, 'Thi xa')

                break

        if unify_ward != None:
            if len(unify_ward.split(' ')) < 2:
                unify_ward = None

        return unify_ward

    def UnifyWardFTel(dict_location, ward, district, city):
        unify_ward = UnifyWardFTel1(dict_location, ward, district, city)

        if unify_ward == None:
            unify_ward = UnifyWardFTel2(dict_location, ward, district)

        if unify_ward == None:
            unify_ward = UnifyWardFTel3(dict_location, ward, district)

        if unify_ward == None:
            unify_ward = UnifyWardFTel4(ward)

        return unify_ward

    stats_ward = profile_internet.groupby(by=['ward', 'district', 'city'],
                                      dropna=False)['contract'].agg(num_customer='count').reset_index()
    stats_ward.loc[stats_ward['ward'].isna(), 'ward'] = None
    stats_ward.loc[stats_ward['district'].isna(), 'district'] = None
    stats_ward.loc[stats_ward['city'].isna(), 'city'] = None

    stats_ward['unify_ward'] = stats_ward.apply(lambda x:
                                                UnifyWardFTel(
                                                    dict_location, x.ward, x.district, x.city),
                                                axis=1)

    profile_internet = profile_internet.merge(
        stats_ward[['ward', 'district', 'city', 'unify_ward']],
        how='left', on=['ward', 'district', 'city']
    )
    profile_internet['ward'] = profile_internet['unify_ward']
    profile_internet = profile_internet.drop(columns=['unify_ward'])

    # unit_address
    columns = ['street']
    profile_internet['unit_address'] = profile_internet[columns].fillna('').agg(' '.join, axis=1).str.replace(
        '(?<![a-zA-Z0-9]),', '', regex=True).str.replace('-(?![a-zA-Z0-9])', '', regex=True)
    profile_internet['unit_address'] = profile_internet['unit_address'].str.strip(
    ).replace(dict_trash)
    profile_internet['unit_address'] = profile_internet['unit_address'].str.title()

    # full_address
    columns = ['unit_address', 'ward', 'district', 'city']
    profile_internet['address'] = None
    profile_internet['address'] = profile_internet[columns].fillna('').agg(', '.join, axis=1).str.replace(
        '(?<![a-zA-Z0-9]),', '', regex=True).str.replace('-(?![a-zA-Z0-9])', '', regex=True)
    profile_internet['address'] = profile_internet['address'].str.strip().replace(
        dict_trash)
    profile_internet = profile_internet.drop(columns=['street'])

    # add info
    print(">>> Filtering out info")
    columns = ['contract', 'phone_raw', 'phone', 'is_phone_valid',
               'email_raw', 'email', 'is_email_valid',
               'name', 'pronoun', 'is_full_name', 'gender',
               'birthday', 'customer_type',  # 'customer_type_detail',
               'address', 'unit_address', 'ward', 'district', 'city', 'source']
    profile_internet = profile_internet[columns]
    profile_internet = profile_internet.rename(columns={'contract': 'uid'})

    # Create contract_phone_ftel
    profile_internet.loc[
        (profile_internet['uid'].notna())
        & (profile_internet['phone'].notna()),
        'contract_phone_ftel'
    ] = profile_internet['uid'] + '-' + profile_internet['phone']
    profile_internet.loc[
        (profile_internet['uid'].notna())
        & (profile_internet['phone'].isna()),
        'contract_phone_ftel'
    ] = profile_internet['uid']

    # Fill 'Ca nhan'
    profile_internet.loc[
        (profile_internet['name'].notna())
        & (profile_internet['customer_type'].isna()),
        'customer_type'
    ] = 'Ca nhan'

    # return
    return profile_internet

# function update profile (unify)


def UpdateUnifyFtel(
    now_str: str,
    n_cores: int = 1
):
    # VARIABLES
    f_group = 'ftel_internet'
    yesterday_str = (datetime.strptime(now_str, '%Y-%m-%d') -
                     timedelta(days=1)).strftime('%Y-%m-%d')

    # load profile (yesterday, now)
    print(">>> Loading today and yesterday profile")
    info_columns = ['uid', 'phone', 'email', 'name', 'birthday',
                    'address', 'street', 'ward', 'district', 'city', 'source']
    now_profile = pd.read_parquet(
        f'{CENTRALIZE_PATH}/{f_group}.parquet/d={now_str}',
        filesystem=hdfs, columns=info_columns
    )
    yesterday_profile = pd.read_parquet(
        f'{CENTRALIZE_PATH}/{f_group}.parquet/d={yesterday_str}',
        filesystem=hdfs, columns=info_columns
    )

    # get profile change/new
    print(">>> Filtering new profile")
    difference_profile = get_difference_data(now_profile, yesterday_profile)
    print(f"Number of new profile {difference_profile.shape}")

    # update profile
    profile_unify = pd.read_parquet(
        f'{PREPROCESS_PATH}/{f_group}.parquet/d={yesterday_str}',
        filesystem=hdfs
    )
    if not difference_profile.empty:
        # get profile unify (old + new)
        new_profile_unify = UnifyFtel(difference_profile, n_cores=n_cores)

        # synthetic profile
        profile_unify = pd.concat(
            [new_profile_unify, profile_unify],
            ignore_index=True
        )

    # arrange columns
    print(">>> Re-Arranging Columns")
    columns = [
        'uid', 'phone_raw', 'phone', 'is_phone_valid',
        'email_raw', 'email', 'is_email_valid',
        'name', 'pronoun', 'is_full_name', 'gender',
        'birthday', 'customer_type',  # 'customer_type_detail',
        'address', 'unit_address', 'ward', 'district', 'city',
        'source', 'contract_phone_ftel'
    ]
    profile_unify = profile_unify[columns].copy()
    profile_unify['birthday'] = profile_unify['birthday'].astype(str)
    profile_unify = profile_unify.drop_duplicates(
        subset=['uid', 'phone_raw', 'email_raw'], keep='first')

    # Type casting for saving
    print(">>> Process casting columns...")
    profile_unify['uid'] = profile_unify['uid'].astype(str)
    profile_unify['birthday'] = profile_unify['birthday'].astype('datetime64[s]')

    # skip name (multi)
    profile_unify.loc[profile_unify['source'] == 'multi', 'name'] = None

    # save
    print(f'Checking {f_group} data for {now_str}...')
    f_group_path = f'{PREPROCESS_PATH}/{f_group}.parquet'
    proc = subprocess.Popen(['hdfs', 'dfs', '-test', '-e', f_group_path + f'/d={now_str}'])
    proc.communicate()
    if proc.returncode == 0:
        print("Data already existed, Removing...")
        subprocess.run(['hdfs', 'dfs', '-rm', '-r', f_group_path + f'/d={now_str}'])

    profile_unify['d'] = now_str
    profile_unify.to_parquet(
        f_group_path,
        filesystem=hdfs, index=False,
        partition_cols='d',
        coerce_timestamps='us',
        allow_truncated_timestamps=True
    )


if __name__ == '__main__':

    DAY = sys.argv[1]
    UpdateUnifyFtel(DAY, n_cores=5)
