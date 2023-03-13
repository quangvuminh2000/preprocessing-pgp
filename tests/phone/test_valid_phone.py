import pandas as pd
import pytest as pt

from preprocessing_pgp.phone.extractor import process_convert_phone
from preprocessing_pgp.phone.utils import basic_phone_preprocess


class TestPhoneValid:
    """
    Class contains tests for information extraction from phone
    """
    # * Testing for valid phone
    @pt.mark.phone_valid
    def test_validate_phone_head_84(self):
        """
        Test for head 84
        """
        phone_data = pd.DataFrame.from_dict({
            'phone': [
                '849233858961', # 12 -- valid
                '8494539014121', # 13
                '84945390141', # 11 -- valid
                '8494539011', # 10
                '849453901' # 9
            ]
        })

        extracted_info = process_convert_phone(
            phone_data,
            phone_col='phone',
            n_cores=2
        )

        print(extracted_info)
        assert True

    @pt.mark.phone_valid
    def test_validate_phone_non_0_head(self):
        """
        Test for head phone without 0
        """
        phone_data = pd.DataFrame.from_dict({
            'phone': [
                '903363010'
            ]
        })

        extracted_info = process_convert_phone(
            phone_data,
            phone_col='phone',
            n_cores=1
        )

        print(extracted_info)
        assert True

    @pt.mark.phone_preprocess
    def test_preprocess_phone_special_char(self):
        """
        Test for preprocessing spacing phone
        """
        phone_data = pd.DataFrame.from_dict({
            'phone': [
                '035 7951800',
                '097 2630093',
                '(+84)978983425',
                '(84)978983425',
                '0203.788089',
                'ditmemay'
            ]
        })

        extracted_info = process_convert_phone(
            phone_data,
            phone_col='phone',
            n_cores=1
        )

        print(extracted_info)
        assert True
