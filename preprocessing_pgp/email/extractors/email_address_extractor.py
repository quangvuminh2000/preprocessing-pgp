"""
Module to extract address from email using rule-based
"""

import re
import pandas as pd

from preprocessing_pgp.address.const import LOCATION_ENRICH_DICT


class EmailAddressExtractor:
    """
    Class contains logic to extract address from email
    """

    def __init__(self):
        self.__generate_norm_dict()
        self.norm_regex = '|'.join(self.location_norm_dict.keys())

    def __generate_norm_dict(self):
        if hasattr(self, 'location_norm_dict'):
            return

        location_norm = LOCATION_ENRICH_DICT.copy()

        city_mask = LOCATION_ENRICH_DICT['lv1_norm'].str.contains(r'\btinh')

        location_norm.loc[
            city_mask,
            'norm_city'
        ] = location_norm.loc[
            city_mask,
            'lv1_norm'
        ].str.extract(r'\s(.*)')[0]\
            .str.replace(r'\s+', '', regex=True)\
            .str.replace('-', '', regex=False)

        location_norm.loc[
            ~city_mask,
            'norm_city'
        ] = location_norm.loc[
            ~city_mask,
            'lv1_norm'
        ].str.extract(r'\s\S*\s(.*)')[0]\
            .str.replace(r'\s+', '', regex=True)\
            .str.replace('-', '', regex=False)

        self.location_norm_dict = location_norm.set_index('norm_city')[
            'lv1'].to_dict()
        # * Additional cities
        self.location_norm_dict['vungtau'] = self.location_norm_dict['bariavungtau']
        self.location_norm_dict['baria'] = self.location_norm_dict['bariavungtau']

    def _get_address(
        self,
        email_name: str
    ) -> str:
        """
        Get Level 1 address from email name
        """
        matches = re.findall(self.norm_regex, email_name)

        if len(matches) == 0:
            return None

        norm_city = matches[0]
        address = self.location_norm_dict.get(norm_city, None)

        return address

    def extract_address(
        self,
        data: pd.DataFrame,
        email_name_col: str = 'email_name'
    ) -> pd.DataFrame:
        """
        Extract address from email name if possible

        Parameters
        ----------
        data : pd.DataFrame
            The input data contains an email_name column
        email_name_col : str, optional
            The name of the column contains the email name, by default 'email_name'

        Returns
        -------
        pd.DataFrame
            Data with additional column for extracted phone:
            * `address_extracted` : Extracted address (city-level 1) from email name
        """

        # * Using regex to search for phone
        data['address_extracted'] =\
            data[email_name_col].apply(self._get_address)

        return data
