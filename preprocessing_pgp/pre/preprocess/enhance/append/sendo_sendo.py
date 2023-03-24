import subprocess
import sys
from datetime import datetime, timedelta

import pandas as pd

sys.path.append("/bigdata/fdp/cdp/source/core_profile/preprocess/utils")
# from enhance_profile import enhance_common_profile
from const import CENTRALIZE_PATH, PREPROCESS_PATH, hdfs

# from preprocess_profile import (
#     remove_same_username_email,
#     extracting_pronoun_from_name
# )
from filter_profile import get_difference_data

# function get profile change/new


# function unify profile
def UnifySendo(profile_sendo: pd.DataFrame, n_cores: int = 1):
    # * Enhancing common profile
    #     profile_sendo = enhance_common_profile(
    #         profile_sendo,
    #         n_cores=n_cores
    #     )

    #     profile_sendo = profile_sendo.rename(columns={'gender_enrich': 'gender'})

    # drop name is username_email
    #     print(">>> Extra Cleansing Name")
    #     profile_sendo = remove_same_username_email(
    #         profile_sendo,
    #         name_col='name',
    #         email_col='email'
    #     )

    #     # clean name, extract pronoun
    #     condition_name = (profile_sendo['customer_type'] == 'Ca nhan')\
    #         & (profile_sendo['name'].notna())
    #     profile_sendo = extracting_pronoun_from_name(
    #         profile_sendo,
    #         condition=condition_name,
    #         name_col='name',
    #     )

    #     # is full name
    #     print(">>> Checking Full Name")
    #     profile_sendo.loc[profile_sendo['last_name'].notna() & profile_sendo['first_name'].notna(), 'is_full_name'] = True
    #     profile_sendo['is_full_name'] = profile_sendo['is_full_name'].fillna(False)
    #     profile_sendo = profile_sendo.drop(columns=['last_name', 'middle_name', 'first_name'])

    # spare unit_address
    #     print(">>> Processing Address")
    #     def SparseUnitAddress(address, ward, district, city):
    #         result = address.title()

    #         if ward != None:
    #             ward = ward.title()

    #             if 'Xa' in ward:
    #                 for key in ['Xa ', 'Xa', ',', '.', '']:
    #                     key_ward = ward.replace('Xa ', key).strip()
    #                     result = result.split(key_ward)[0]
    #             if 'Phuong' in ward:
    #                 for key in ['Phuong ', 'Phuong', 'P', 'F', 'P.', 'F.', 'F ', 'P ', ',', '.', '']:
    #                     key_ward = ward.replace('Phuong ', key).strip().replace('0', '')
    #                     if key_ward.isdigit():
    #                         continue
    #                     result = result.split(key_ward)[0]
    #             elif 'Thi Tran' in ward:
    #                 for key in ['Thi Tran ', 'Thi Tran', 'Tt ', 'Tt.', ',', '.', '']:
    #                     key_ward = ward.replace('Thi Tran ', key).strip()
    #                     result = result.split(key_ward)[0]

    #         # district
    #         if district != None:
    #             district = district.title()

    #             if 'Huyen' in district:
    #                 for key in ['Huyen ', 'Huyen', 'H ', 'H.', ',', '.', '']:
    #                     key_district = district.replace('Huyen ', key).strip()
    #                     result = result.split(key_district)[0]
    #             elif 'Thi Xa' in district:
    #                 for key in ['Thi Xa ', 'Thi xa', 'Tx ', 'Tx.', ',', '.', '']:
    #                     key_district = district.replace('Thi Xa ', key).strip()
    #                     result = result.split(key_district)[0]
    #             elif 'Quan' in district:
    #                 for key in ['Quan ', 'Quan', 'Q', 'Q.', ',', '.', '']:
    #                     key_district = district.replace('Quan ', key).strip().replace('0', '')
    #                     if key_district.isdigit():
    #                         continue
    #                     result = result.split(key_district)[0]
    #             elif 'Thanh Pho' in district:
    #                 for key in ['Thanh Pho ', 'Thanh Pho', 'Tp ', 'Tp.', ',', '.', '']:
    #                     key_district = district.replace('Thanh Pho ', key).strip()
    #                     result = result.split(key_district)[0]

    #         # city
    #         if city != None:
    #             city = city.title()
    #             for key in ['Tinh ', 'Tinh', 'Thanh Pho ', 'Thanh Pho', 'T.', 'Tp', 'Tp.', ',', '.', '']:
    #                 key_city = (key + city).strip()
    #                 result = result.split(key_city)[0]

    #         # Normalize
    #         result = result.strip()
    #         if result in [None, '']:
    #             result = None
    #         else:
    #             result = result[:-1].strip() if (result[-1] in [',', '.']) else result

    #         # Fix UnitAdress is FullAddress
    #         if (result != None) & (district != None) & (city != None):
    #             have_district = False
    #             for key_district in [' Huyen ', ' Thi Xa ', ' Quan ', ' Thanh Pho ']:
    #                 if key_district.lower() in result.lower():
    #                     have_district = True
    #                     break

    #             have_city = False
    #             for key_city in [' Tinh ', ' Thanh Pho ']:
    #                 if key_city.lower() in result.lower():
    #                     have_city = True
    #                     break

    #             if (have_district == True) & (have_city == True):
    #                 result = None

    #         if (result != None) & (district != None):
    #             for key_district in [' Huyen ', ' Thi Xa ', ' Quan ', ' Thanh Pho ']:
    #                 if key_district.lower() in result.lower():
    #                     result = result.split(',')[0].strip()
    #                     if len(result.split(' ')) > 5:
    #                         result = None
    #                     break

    #         return result

    #     def UltimatelyUnescape(s: str) -> str:
    #         unescaped = ""
    #         while unescaped != s:
    #             s = html.unescape(s)
    #             unescaped = html.unescape(s)

    #         return s

    #     condition_address = profile_sendo['address'].notna()
    #     profile_sendo.loc[condition_address, 'address'] = profile_sendo.loc[condition_address, 'address'].str.lower().apply(UltimatelyUnescape).str.title()
    #     profile_sendo['address'] = profile_sendo['address'].replace({'[0-9]{6,}': ''}, regex=True)

    #     profile_sendo.loc[condition_address, 'unit_address'] = profile_sendo[condition_address].apply(lambda x: SparseUnitAddress(x.address, x.ward, x.district, x.city), axis=1)
    #     profile_sendo.loc[profile_sendo['unit_address'].isna(), 'unit_address'] = None
    #     # profile_sendo['unit_address'] = profile_sendo['unit_address'].str.replace('. ', ', ').str.replace('; ', ',')

    #     # unify city
    #     profile_sendo['city'] = profile_sendo['city'].replace({'Ba Ria-Vung Tau': 'Vung Tau',
    #                                                            'Dak Nong': 'Dac Nong',
    #                                                            'Bac Kan': 'Bac Can'})

    #     norm_sendo_city = pd.read_parquet('/data/fpt/ftel/cads/dep_solution/user/namdp11/scross_fill/runner/refactor/material/ftel_provinces.parquet',
    #                                       filesystem=hdfs)
    #     norm_sendo_city.columns = ['city', 'norm_city']
    #     profile_sendo = pd.merge(
    #         profile_sendo.set_index('city'),
    #         norm_sendo_city.set_index('city'),
    #         left_index=True, right_index=True,
    #         how='left',
    #         sort=False
    #     ).reset_index()
    #     profile_sendo['city'] = profile_sendo['norm_city']
    #     profile_sendo = profile_sendo.drop(columns=['norm_city'])

    #     # unify district
    #     def UnifyDistrictSendo(dict_location, district, city):
    #         if district == None:
    #             return None

    #         district = unidecode(district)
    #         location = dict_location[['district', 'city']].drop_duplicates().copy()

    #         if city != None:
    #             location = location[location['city'] == city]
    #         if district in ['Quan Thu Duc', 'Quan 9', 'Quan 2', 'Thanh pho Thu Duc']:
    #             return district

    #         temp_district = district.title().replace('Huyen ', '').replace('Thi Xa ', '').replace('Thanh Pho ', '').strip()
    #         location['district_temp'] = location['district'].str.title().replace({'Huyen ': '',
    #                                                                               'Thi Xa': '',
    #                                                                               'Thanh Pho ': '',
    #                                                                               'Quan ': ''}, regex=True).str.strip()
    #         location['similar'] = location['district_temp'].apply(lambda x: SequenceMatcher(None, temp_district.lower(), x.lower()).ratio())

    #         location = location[(location['similar'] == location['similar'].max()) &
    #                             (location['similar'] >= 0.8)]

    #         unify_district = None
    #         if location.empty == False:
    #             unify_district = location['district'].values[0]

    #         if unify_district == None:
    #             unify_district = district

    #         return unify_district

    #     stats_district_sendo = profile_sendo.groupby(by=['district', 'city'])['uid'].agg(num_customer='count').reset_index()
    #     dict_district = dict_location[['district', 'city']].drop_duplicates().copy()
    #     dict_district['new_district'] = dict_district['district']
    #     stats_district_sendo = stats_district_sendo.merge(dict_district, how='left', on=['district', 'city'])

    #     condition_district = stats_district_sendo['new_district'].isna()
    #     stats_district_sendo.loc[condition_district, 'new_district'] = stats_district_sendo[condition_district].apply(lambda x: UnifyDistrictSendo(dict_location, x.district, x.city), axis=1)
    #     stats_district_sendo = stats_district_sendo.drop(columns=['num_customer'])

    #     profile_sendo = profile_sendo.merge(stats_district_sendo, how='left', on=['district', 'city'])
    #     profile_sendo['district'] = profile_sendo['new_district']
    #     profile_sendo = profile_sendo.drop(columns=['new_district'])

    #     # unify ward
    #     def UnifyWardSendo(dict_location, ward, district, city):
    #         if ward == None:
    #             return None

    #         ward = unidecode(ward).title()
    #         location = dict_location[['ward', 'district', 'city']].drop_duplicates().copy()

    #         if city != None:
    #             location = location[location['city'] == city]
    #         if district != None:
    #             if district in ['Quan Thu Duc', 'Quan 9', 'Quan 2']:
    #                 district = 'Thanh pho Thu Duc'
    #             location = location[location['district'] == district]

    #         temp_ward = ward.title().replace('Xa ', '').replace('Phuong ', '').replace('Thi Tran ', '').replace('0', '').strip()
    #         location['ward_temp'] = location['ward'].str.title().replace({'Xa ': '',
    #                                                                       'Phuong ': '',
    #                                                                       'Thi Tran ': '',
    #                                                                       '0': ''}, regex=True).str.strip()
    #         location['similar'] = location['ward_temp'].apply(lambda x: SequenceMatcher(None, temp_ward.lower(), x.lower()).ratio())

    #         location = location[(location['similar'] == location['similar'].max()) &
    #                             (location['similar'] >= 0.8)]

    #         unify_ward = None
    #         if location.empty == False:
    #             unify_ward = location['ward'].values[0]

    #         if unify_ward == None:
    #             unify_ward = ward

    #         return unify_ward

    #     stats_ward_sendo = profile_sendo.groupby(by=['ward', 'district', 'city'])['uid'].agg(num_customer='count').reset_index()

    #     dict_ward = dict_location[['ward', 'district', 'city']].drop_duplicates().copy()
    #     dict_ward['new_ward'] = dict_ward['ward']
    #     stats_ward_sendo = stats_ward_sendo.merge(dict_ward, how='left', on=['ward', 'district', 'city'])

    #     condition_ward = stats_ward_sendo['new_ward'].isna()
    #     stats_ward_sendo.loc[condition_ward, 'new_ward'] = stats_ward_sendo[condition_ward].apply(lambda x: UnifyWardSendo(dict_location, x.ward, x.district, x.city), axis=1)
    #     stats_ward_sendo = stats_ward_sendo.drop(columns=['num_customer'])

    #     profile_sendo = profile_sendo.merge(stats_ward_sendo, how='left', on=['ward', 'district', 'city'])
    #     profile_sendo['ward'] = profile_sendo['new_ward']
    #     profile_sendo = profile_sendo.drop(columns=['new_ward'])

    #     condition_ward_error = profile_sendo['ward'].str.contains('0[1-9]', case=False, na=False)
    #     profile_sendo.loc[condition_ward_error, 'ward'] = profile_sendo.loc[condition_ward_error, 'ward'].str.replace('0', '')

    #     # full_address
    #     profile_sendo['address'] = None
    #     columns = ['unit_address', 'ward', 'district', 'city']
    #     profile_sendo['address'] = profile_sendo[columns].fillna('').agg(', '.join, axis=1).str.replace('(?<![a-zA-Z0-9]),', '', regex=True).str.replace('-(?![a-zA-Z0-9])', '', regex=True)

    # add info
    print(">>> Adding Temp Info")
    # profile_sendo['birthday'] = None
    profile_sendo[
        [
            "source_address",
            "source_city",
            "source_district",
            "source_ward",
            "source_street",
        ]
    ] = None
    columns = [
        "uid",
        "phone",
        "email",
        "card",
        "name",
        "gender",
        "birthday",
        "age",
        "customer_type",
        "address",
        "city",
        "district",
        "ward",
        "street",
        "source_address",
        "source_city",
        "source_district",
        "source_ward",
        "source_street",
        "source",
    ]
    profile_sendo = profile_sendo[columns]

    # Fill 'Ca nhan'
    # profile_sendo.loc[profile_sendo['name'].notna() & profile_sendo['customer_type'].isna(), 'customer_type'] = 'Ca nhan'

    # Create id_phone_sendo
    # profile_sendo.loc[
    #     (profile_sendo['uid'].notna())
    #     & (profile_sendo['phone'].notna()),
    #     'id_phone_sendo'
    # ] = profile_sendo['uid'] + '-' + profile_sendo['phone']
    # profile_sendo.loc[
    #     (profile_sendo['uid'].notna())
    #     & (profile_sendo['phone'].isna()),
    #     'id_phone_sendo'
    # ] = profile_sendo['uid']

    # return
    return profile_sendo


# function update profile (unify)
def UpdateUnifySendo(now_str: str, n_cores: int = 1):
    # VARIABLES
    f_group = "sendo_sendo"
    yesterday_str = (
        datetime.strptime(now_str, "%Y-%m-%d") - timedelta(days=1)
    ).strftime("%Y-%m-%d")

    # load profile (yesterday, now)
    print(">>> Loading today and yesterday profile")
    info_columns = [
        "uid",
        "phone",
        "email",
        "card",
        "name",
        "gender",
        "birthday",
        "age",
        "customer_type",
        "address",
        "city",
        "district",
        "ward",
        "street",
        "source",
    ]
    now_profile = pd.read_parquet(
        f"{CENTRALIZE_PATH}/{f_group}.parquet/d={now_str}",
        filesystem=hdfs,
        columns=info_columns,
    )
    yesterday_profile = pd.read_parquet(
        f"{CENTRALIZE_PATH}/{f_group}.parquet/d={yesterday_str}",
        filesystem=hdfs,
        columns=info_columns,
    )

    # get profile change/new
    print(">>> Filtering new profile")
    difference_profile = get_difference_data(now_profile, yesterday_profile)
    print(f"Number of new profile {difference_profile.shape}")

    # update profile
    profile_unify = pd.read_parquet(
        f"{PREPROCESS_PATH}/{f_group}.parquet/d={yesterday_str}",
        filesystem=hdfs,
    )
    if not difference_profile.empty:
        # get profile unify (old + new)
        new_profile_unify = UnifySendo(difference_profile, n_cores=n_cores)

        # synthetic profile
        profile_unify = pd.concat(
            [new_profile_unify, profile_unify], ignore_index=True
        )

    # arrange columns
    print(">>> Re-Arranging Columns")
    columns = [
        "uid",
        "phone",
        "email",
        "card",
        "name",
        "gender",
        "birthday",
        "age",
        "customer_type",
        "address",
        "city",
        "district",
        "ward",
        "street",
        "source_address",
        "source_city",
        "source_district",
        "source_ward",
        "source_street",
        "source",
    ]

    profile_unify = profile_unify[columns]
    # profile_unify['is_phone_valid'] = profile_unify['is_phone_valid'].fillna(False)
    # profile_unify['is_email_valid'] = profile_unify['is_email_valid'].fillna(False)
    # profile_unify = profile_unify.drop_duplicates(subset=['uid', 'phone_raw', 'email_raw'], keep='first')

    # Type casting for saving
    print(">>> Process casting columns...")
    # profile_unify['uid'] = profile_unify['uid'].astype(str)
    profile_unify["birthday"] = profile_unify["birthday"].astype(
        "datetime64[s]"
    )

    # save
    print(f"Checking {f_group} data for {now_str}...")
    f_group_path = f"{PREPROCESS_PATH}/{f_group}.parquet"
    proc = subprocess.Popen(
        ["hdfs", "dfs", "-test", "-e", f_group_path + f"/d={now_str}"]
    )
    proc.communicate()
    if proc.returncode == 0:
        print("Data already existed, Removing...")
        subprocess.run(
            ["hdfs", "dfs", "-rm", "-r", f_group_path + f"/d={now_str}"]
        )

    profile_unify["d"] = now_str
    profile_unify.to_parquet(
        f_group_path,
        filesystem=hdfs,
        index=False,
        partition_cols="d",
        coerce_timestamps="us",
        allow_truncated_timestamps=True,
    )


if __name__ == "__main__":
    DAY = sys.argv[1]
    UpdateUnifySendo(DAY, n_cores=5)
