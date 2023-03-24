import subprocess
import sys

import pandas as pd

sys.path.append("/bigdata/fdp/cdp/source/core_profile/preprocess/utils")
from const import CENTRALIZE_PATH, PREPROCESS_PATH, hdfs


def UnifyFtel(date_str: str, n_cores: int = 1):
    # VARIABLES
    f_group = "ftel_internet"

    # load profile ftel
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
    profile_internet = pd.read_parquet(
        f"{CENTRALIZE_PATH}/{f_group}.parquet/d={date_str}",
        filesystem=hdfs,
        columns=info_columns,
    )

    # print(">>> Cleansing profile")
    # profile_ftel = cleansing_profile_name(
    #     profile_ftel,
    #     name_col='name',
    #     n_cores=n_cores
    # )
    # profile_ftel.rename(columns={
    #     'email': 'email_raw',
    #     'phone': 'phone_raw',
    #     'name': 'raw_name'
    # }, inplace=True)

    # # * Loadding dictionary
    # print(">>> Loading dictionaries")
    # profile_phones = profile_ftel['phone_raw'].drop_duplicates().dropna()
    # profile_emails = profile_ftel['email_raw'].drop_duplicates().dropna()
    # profile_names = profile_ftel['raw_name'].drop_duplicates().dropna()

    # # phone, email (valid)
    # valid_phone = pd.read_parquet(
    #     f'{UTILS_PATH}/valid_phone_latest.parquet',
    #     filters=[('phone_raw', 'in', profile_phones)],
    #     filesystem=hdfs,
    #     columns=['phone_raw', 'phone', 'is_phone_valid']
    # )
    # valid_email = pd.read_parquet(
    #     f'{UTILS_PATH}/valid_email_latest.parquet',
    #     filters=[('email_raw', 'in', profile_emails)],
    #     filesystem=hdfs,
    #     columns=['email_raw', 'email', 'is_email_valid']
    # )
    # dict_name_lst = pd.read_parquet(
    #     f'{UTILS_PATH}/dict_name_latest.parquet',
    #     filters=[('raw_name', 'in', profile_names)],
    #     filesystem=hdfs,
    #     columns=[
    #         'raw_name', 'enrich_name',
    #         'last_name', 'middle_name', 'first_name',
    #         'gender'
    #     ]
    # )

    # info
    # merge get phone, email (valid)
    # print(">>> Merging phone, email, name")
    # profile_ftel = pd.merge(
    #     profile_ftel.set_index('phone_raw'),
    #     valid_phone.set_index('phone_raw'),
    #     left_index=True, right_index=True,
    #     how='left',
    #     sort=False
    # ).reset_index(drop=False)

    # profile_ftel = pd.merge(
    #     profile_ftel.set_index('email_raw'),
    #     valid_email.set_index('email_raw'),
    #     left_index=True, right_index=True,
    #     how='left',
    #     sort=False
    # ).reset_index(drop=False)

    # profile_ftel = pd.merge(
    #     profile_ftel.set_index('raw_name'),
    #     dict_name_lst.set_index('raw_name'),
    #     left_index=True, right_index=True,
    #     how='left',
    #     sort=False
    # ).rename(columns={
    #     'enrich_name': 'name'
    # }).reset_index(drop=False)

    # # Refilling info
    # cant_predict_name_mask = profile_ftel['name'].isna()
    # profile_ftel.loc[
    #     cant_predict_name_mask,
    #     'name'
    # ] = profile_ftel.loc[
    #     cant_predict_name_mask,
    #     'raw_name'
    # ]
    # profile_ftel['name'] = profile_ftel['name'].replace(dict_trash)

    # # load datapay => customer type
    # ds_contract = pd.read_parquet('/data/fpt/ftel/isc/dwh/ds_contract.parquet',
    #                               columns=['contract', 'net_customer_type'],
    #                               filesystem=hdfs).drop_duplicates(subset=['contract'], keep='last')
    # ds_contract.columns = ['contract', 'datapay_customer_type']
    # profile_ftel = profile_ftel.merge(ds_contract, how='left', on='contract')
    # profile_ftel.loc[profile_ftel['source'] == 'multi', 'datapay_customer_type'] = None
    # profile_ftel.loc[profile_ftel['source'] == 'multi', 'city'] = None

    # birthday
    print(">>> Processing Birthday")
    condition_birthday = profile_internet["birthday"].notna()
    profile_internet.loc[condition_birthday, "birthday"] = pd.to_datetime(
        profile_internet[condition_birthday]["birthday"].astype(str),
        errors="coerce",
    ).dt.strftime("%d/%m/%Y")

    # customer_type
    # print(">>> Extracting customer type")
    # profile_ftel = process_extract_name_type(
    #     profile_ftel,
    #     name_col='name',
    #     n_cores=n_cores,
    #     logging_info=False
    # )
    # profile_ftel['customer_type'] =\
    #     profile_ftel['customer_type'].map({
    #         'customer': 'Ca nhan',
    #         'company': 'Cong ty',
    #         'medical': 'Benh vien - Phong kham',
    #         'edu': 'Giao duc',
    #         'biz': 'Ho kinh doanh'
    #     })

    # profile_ftel.loc[profile_ftel['customer_type'].isna(), 'customer_type'] =  profile_ftel['datapay_customer_type']
    # profile_ftel = profile_ftel.drop(columns=['datapay_customer_type'])

    # # drop name is username_email
    # print(">>> Extra Cleansing Name")
    # profile_ftel = remove_same_username_email(
    #     profile_ftel,
    #     name_col='name',
    #     email_col='email'
    # )

    # # clean name
    # condition_name = (profile_ftel['customer_type'] == 'customer')\
    #     & (profile_ftel['name'].notna())
    # profile_ftel = extracting_pronoun_from_name(
    #     profile_ftel,
    #     condition=condition_name,
    #     name_col='name',
    # )

    # # is full name
    # print(">>> Checking Full Name")
    # profile_ftel.loc[profile_ftel['last_name'].notna(
    # ) & profile_ftel['first_name'].notna(), 'is_full_name'] = True
    # profile_ftel['is_full_name'] = profile_ftel['is_full_name'].fillna(False)
    # profile_ftel = profile_ftel.drop(
    #     columns=['last_name', 'middle_name', 'first_name'])

    # # unify location
    # print(">>> Processing Address")
    # norm_ftel_city = pd.read_parquet('/data/fpt/ftel/cads/dep_solution/user/namdp11/scross_fill/runner/refactor/material/ftel_provinces.parquet',
    #                                  filesystem=hdfs)
    # norm_ftel_city.columns = ['city', 'norm_city']

    # norm_ftel_district = pd.read_parquet('/data/fpt/ftel/cads/dep_solution/user/namdp11/scross_fill/runner/refactor/material/ftel_districts.parquet',
    #                                      filesystem=hdfs)
    # norm_ftel_district.columns = ['district', 'norm_city', 'norm_district', 'new_norm_district']

    # ## update miss district
    # district_update = list(set(profile_ftel['district']) - set(norm_ftel_district['district']))
    # location_dict = pd.read_parquet('/data/fpt/ftel/cads/dep_solution/user/namdp11/scross_fill/runner/refactor/material/location_dict.parquet',
    #                                 filesystem=hdfs)
    # location_dict.columns = ['city', 'district', 'norm_city', 'norm_district']
    # district_list = list(location_dict['norm_district'].unique())

    # def fix_miss_district(district_list, district):
    #     df = pd.DataFrame()
    #     df['district'] = district_list
    #     df['similar'] = df['district'].apply(lambda x: SequenceMatcher(None, district.lower(), x.lower()).ratio())

    #     df = df[df.similar == df.similar.max()]
    #     result = df['district'].values[0]
    #     return result

    # miss_district = {}
    # for district in district_update:
    #     if district != None:
    #         result = fix_miss_district(district_list, district)
    #         miss_district[district] = result

    # ## merge city, district
    # profile_ftel = profile_ftel.merge(norm_ftel_city, how='left', on='city')
    # profile_ftel = profile_ftel.merge(norm_ftel_district[['district', 'norm_city', 'new_norm_district']],
    #                                   how='left', on=['district', 'norm_city'])

    # ## fix bugs location
    # profile_ftel.loc[profile_ftel['contract'].notna() &
    #                  profile_ftel['phone'].notna(), 'contract_phone_ftel'] = profile_ftel['phone'] + '-' + profile_ftel['contract']
    # profile_ftel.loc[profile_ftel['contract'].notna() &
    #                  profile_ftel['phone'].isna(), 'contract_phone_ftel'] = profile_ftel['contract']

    # bug_location_ftel = profile_ftel[profile_ftel['norm_city'].isna() | profile_ftel['new_norm_district'].isna()]
    # bug_location_ftel1 = bug_location_ftel[bug_location_ftel['norm_city'].notna()]
    # bug_location_ftel2 = bug_location_ftel[~bug_location_ftel['contract_phone_ftel'].isin(bug_location_ftel1['contract_phone_ftel'])]

    # bug_location_ftel1 = bug_location_ftel1.drop(columns=['new_norm_district'])
    # bug_location_ftel1 = bug_location_ftel1.merge(norm_ftel_district[['district', 'new_norm_district']],
    #                                               how='left', on=['district'])

    # bug_location_ftel2 = bug_location_ftel2.drop(columns=['norm_city', 'new_norm_district'])
    # bug_location_ftel2 = bug_location_ftel2.merge(norm_ftel_district[['district', 'norm_city', 'new_norm_district']],
    #                                               how='left', on=['district'])

    # bug_location_ftel = bug_location_ftel1.append(bug_location_ftel2, ignore_index=True)

    # profile_ftel = profile_ftel[~profile_ftel['contract_phone_ftel'].isin(bug_location_ftel['contract_phone_ftel'])]
    # profile_ftel = profile_ftel.append(bug_location_ftel, ignore_index=True)

    # profile_ftel.loc[profile_ftel['district'].isin(miss_district.keys()), 'new_norm_district'] = profile_ftel['district'].map(miss_district)

    # profile_ftel['city'] = profile_ftel['norm_city']
    # profile_ftel['district'] = profile_ftel['new_norm_district']
    # profile_ftel = profile_ftel.drop(columns=['norm_city', 'new_norm_district'])

    # ## fix distric-city
    # dict_district = dict_location[['district', 'city']].drop_duplicates().copy()
    # stats_district = dict_district.groupby(by=['district'])['city'].agg(num_city='count').reset_index()
    # stats_district = stats_district[stats_district['num_city'] == 1]
    # dict_district = dict_district[dict_district['district'].isin(stats_district['district'])]
    # dict_district = dict_district.rename(columns={'city': 'new_city'})
    # dict_district = dict_district.append([{'district': 'Quan Thu Duc', 'new_city': 'Thanh pho Ho Chi Minh'},
    #                                       {'district': 'Quan 9', 'new_city': 'Thanh pho Ho Chi Minh'},
    #                                       {'district': 'Quan 2', 'new_city': 'Thanh pho Ho Chi Minh'}],
    #                                      ignore_index=True)

    # profile_ftel = profile_ftel.merge(dict_district, how='left', on='district')
    # profile_ftel.loc[profile_ftel['new_city'].notna(), 'city'] = profile_ftel['new_city']
    # profile_ftel = profile_ftel.drop(columns=['new_city'])

    # ## unify ward
    # def UnifyWardFTel1(dict_location, ward, district, city):
    #     if ward == None:
    #         return None

    #     ward = unidecode(ward)
    #     location = dict_location[['ward', 'district', 'city']].drop_duplicates().copy()

    #     if city != None:
    #         location = location[location['city'] == city]
    #     if district != None:
    #         if district in ['Quan Thu Duc', 'Quan 9', 'Quan 2']:
    #             district = 'Thanh pho Thu Duc'
    #         location = location[location['district'] == district]

    #     location['ward_temp'] = location['ward'].str.replace('Phuong 0', 'Phuong ')

    #     location['similar'] = location['ward_temp'].apply(lambda x: SequenceMatcher(None, ward.lower(), x.lower()).ratio())

    #     location = location[(location['similar'] == location['similar'].max()) &
    #                         (location['similar'] >= 0.8)]

    #     unify_ward = None
    #     if location.empty == False:
    #         unify_ward = location['ward'].values[0]

    #     return unify_ward

    # def UnifyWardFTel2(dict_location, ward, district):
    #     if ward == None:
    #         return None

    #     ward = unidecode(ward)
    #     location = dict_location[['ward', 'district']].drop_duplicates().copy()

    #     ward = ward.title().replace('P.', 'Phuong ').replace('F.', 'Phuong ')
    #     ward = ward.title().replace('T.Tran', 'Thi Tran ').replace('T.T', 'Thi Tran ').replace('Tt', 'Thi Tran ')
    #     ward = ward.title().replace('Tx', 'Thi Xa ')

    #     if district != None:
    #         location = location[location['district'] == district]

    #     location['ward_temp'] = location['ward'].str.replace('Phuong 0', 'Phuong ')

    #     location['similar'] = location['ward_temp'].apply(lambda x: SequenceMatcher(None, ward.lower(), x.lower()).ratio())

    #     location = location[(location['similar'] == location['similar'].max()) &
    #                         (location['similar'] >= 0.8)]

    #     unify_ward = None
    #     if location.empty == False:
    #         unify_ward = location['ward'].values[0]

    #     return unify_ward

    # def UnifyWardFTel3(dict_location, ward, district):
    #     if ward == None:
    #         return None

    #     ward = unidecode(ward)
    #     location = dict_location[['ward', 'district']].drop_duplicates().copy()

    #     if district != None:
    #         location = location[location['district'] == district]

    #     ward = ward.title().replace('P.', '').replace('F.', '')
    #     ward = ward.title().replace('T.Tran', '').replace('T.T', '').replace('Tt', '')
    #     ward = ward.title().replace('Tx', '')

    #     location['ward_temp'] = location['ward'].str.replace('Phuong 0', 'Phuong ')
    #     location['ward_temp'] = location['ward_temp'].replace({'Phuong ': '',
    #                                                            'Xa ': '',
    #                                                            'Thi Tran ': '',
    #                                                            'Thi Xa ': ''}, regex=True)

    #     location['similar'] = location['ward_temp'].apply(lambda x: SequenceMatcher(None, ward.lower(), x.lower()).ratio())

    #     location = location[(location['similar'] == location['similar'].max()) &
    #                         (location['similar'] >= 0.8)]

    #     unify_ward = None
    #     if location.empty == False:
    #         unify_ward = location['ward'].values[0]

    #     return unify_ward

    # def UnifyWardFTel4(ward):
    #     if ward == None:
    #         return None

    #     ward = unidecode(ward)
    #     ward = ward.title()
    #     unify_ward = None
    #     for key in ['Xa', 'Phuong', 'Huyen', 'Thi Tran', 'T.Tran', 'Tt', 'T.T', 'P.', 'F.', 'Thi Xa', 'Tx']:
    #         check_idx = ward.find(key)
    #         if check_idx == 0:
    #             if key in ['Xa', 'Phuong', 'Huyen']:
    #                 unify_ward = ward
    #             elif key in ['Thi Tran', 'T.Tran', 'Tt', 'T.T']:
    #                 unify_ward = ward.replace(key, 'Thi tran')
    #             elif key in ['P.', 'F.']:
    #                 unify_ward = ward.replace(key, 'Phuong ')
    #             elif key in ['Thi Xa', 'Tx']:
    #                 unify_ward = ward.replace(key, 'Thi xa')

    #             break

    #     if unify_ward != None:
    #         if len(unify_ward.split(' ')) < 2:
    #             unify_ward = None

    #     return unify_ward

    # def UnifyWardFTel(dict_location, ward, district, city):
    #     unify_ward = UnifyWardFTel1(dict_location, ward, district, city)

    #     if unify_ward == None:
    #          unify_ward = UnifyWardFTel2(dict_location, ward, district)

    #     if unify_ward == None:
    #          unify_ward = UnifyWardFTel3(dict_location, ward, district)

    #     if unify_ward == None:
    #         unify_ward = UnifyWardFTel4(ward)

    #     return unify_ward

    # stats_ward = profile_ftel.groupby(by=['ward', 'district', 'city'],
    #                                   dropna=False)['contract'].agg(num_customer='count').reset_index()
    # stats_ward.loc[stats_ward['ward'].isna(), 'ward'] = None
    # stats_ward.loc[stats_ward['district'].isna(), 'district'] = None
    # stats_ward.loc[stats_ward['city'].isna(), 'city'] = None

    # stats_ward['unify_ward'] = stats_ward.apply(lambda x:
    #                                             UnifyWardFTel(dict_location, x.ward, x.district, x.city),
    #                                             axis=1)

    # profile_ftel = profile_ftel.merge(stats_ward[['ward', 'district', 'city', 'unify_ward']],
    #                                   how='left', on=['ward', 'district', 'city'])
    # profile_ftel['ward'] = profile_ftel['unify_ward']
    # profile_ftel = profile_ftel.drop(columns=['unify_ward'])

    # ## unit_address
    # columns = ['house_number', 'street']
    # profile_ftel['unit_address'] = profile_ftel[columns].fillna('').agg(' '.join, axis=1).str.replace('(?<![a-zA-Z0-9]),', '', regex=True).str.replace('-(?![a-zA-Z0-9])', '', regex=True)
    # profile_ftel['unit_address'] = profile_ftel['unit_address'].str.strip().replace(dict_trash)
    # profile_ftel['unit_address'] = profile_ftel['unit_address'].str.title()

    # ## full_address
    # columns = ['unit_address', 'ward', 'district', 'city']
    # profile_ftel['address'] = None
    # profile_ftel['address'] = profile_ftel[columns].fillna('').agg(', '.join, axis=1).str.replace('(?<![a-zA-Z0-9]),', '', regex=True).str.replace('-(?![a-zA-Z0-9])', '', regex=True)
    # profile_ftel['address'] = profile_ftel['address'].str.strip().replace(dict_trash)
    # profile_ftel = profile_ftel.drop(columns=['house_number', 'street'])

    # add info
    print(">>> Filtering out info")
    profile_internet[
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
    profile_internet = profile_internet[columns]
    profile_internet["birthday"] = profile_internet["birthday"].astype(
        "datetime64[s]"
    )

    # Fill 'Ca nhan'
    # profile_ftel.loc[
    #     (profile_ftel['name'].notna())
    #     & (profile_ftel['customer_type'].isna()),
    #     'customer_type'
    # ] = 'Ca nhan'

    # Create contract_phone_ftel
    # profile_ftel.loc[
    #     (profile_ftel['uid'].notna())
    #     & (profile_ftel['phone'].notna()),
    #     'contract_phone_ftel'
    # ] = profile_ftel['uid'] + '-' + profile_ftel['phone']
    # profile_ftel.loc[
    #     (profile_ftel['uid'].notna())
    #     & (profile_ftel['phone'].isna()),
    #     'contract_phone_ftel'
    # ] = profile_ftel['uid']

    #     # skip name (multi)
    profile_internet.loc[profile_internet["source"] == "multi", "name"] = None

    # Save
    print(f"Checking {f_group} data for {date_str}...")
    f_group_path = f"{PREPROCESS_PATH}/{f_group}.parquet"
    proc = subprocess.Popen(
        ["hdfs", "dfs", "-test", "-e", f_group_path + f"/d={date_str}"]
    )
    proc.communicate()
    if proc.returncode == 0:
        print("Data already existed, Removing...")
        subprocess.run(
            ["hdfs", "dfs", "-rm", "-r", f_group_path + f"/d={date_str}"]
        )

    profile_internet["d"] = date_str
    profile_internet.to_parquet(
        f"{PREPROCESS_PATH}/{f_group}.parquet",
        filesystem=hdfs,
        index=False,
        partition_cols="d",
        coerce_timestamps="us",
        allow_truncated_timestamps=True,
    )


if __name__ == "__main__":
    DAY = sys.argv[1]
    UnifyFtel(DAY, n_cores=5)
