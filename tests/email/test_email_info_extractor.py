"""
Tests for email info extraction
"""
import math
from pprint import pprint

import pytest as pt
import pandas as pd

from preprocessing_pgp.email.extractors.email_name_extractor import EmailNameExtractor
from preprocessing_pgp.email.extractors.email_yob_extractor import EmailYOBExtractor
from preprocessing_pgp.email.extractors.email_phone_extractor import EmailPhoneExtractor
from preprocessing_pgp.email.extractors.email_address_extractor import EmailAddressExtractor
from preprocessing_pgp.email.validator import process_validate_email
from preprocessing_pgp.email.info_extractor import process_extract_email_info


class TestEmailInfoExtraction:
    """
    Class contains tests for information extraction from email
    """
    # * Testing for extracting all info
    @pt.mark.email2info
    def test_extract_full_info_basic(self):
        """
        Test whether the function can extract for full name in email name
        with basic name
        """
        email_name_data = pd.DataFrame.from_dict({
            'email': [
                'vuminhquang00@gmail.com',
                'nguyenphamanhnguyen1996@fpt.com.vn',
                'nguyenhuynhhuy1997@yahoo.com.vn',
                'lekhactrinh1992@gmail.com',
                'vuminhquang12102000@fpt.com.vn',
                'thanhhien.ht08@gmail.com'
            ]
        })


        extracted_info = process_extract_email_info(
            email_name_data,
            email_col='email',
            n_cores=1
        )
        print(extracted_info)
        print(extracted_info.columns)

        assert True

    # * Testing for names
    @pt.mark.email2name
    def test_extract_full_name_basic(self):
        """
        Test whether the function can extract for full name in email name
        with basic name
        """
        name_extractor = EmailNameExtractor()
        email_name_data = pd.DataFrame.from_dict({
            'email_name': [
                'vuminhquang2000',
                'nguyenphamanhnguyen1996',
                'nguyenhuynhhuy1997',
                'lekhactrinh1992',
                'doanhnghiepfpt1212'
            ]
        })

        extracted_name = name_extractor.extract_username(
            email_name_data,
            email_name_col='email_name'
        )['username_extracted'].tolist()
        pprint(extracted_name, compact=True)

        assert extracted_name == [
            'Vu Minh Quang',
            'Nguyen Pham Anh Nguyen',
            'Nguyen Huynh Huy',
            'Le Khac Trinh',
            'Do Anh Nghiep'
        ]

    @pt.mark.email2name
    def test_extract_full_name_in_mixed_order(self):
        """
        Test whether the function can extract for full name in email name
        with mixed order name
        """
        name_extractor = EmailNameExtractor()
        email_name_data = pd.DataFrame.from_dict({
            'email_name': [
                'quangvuminh2000',
                'nguyennguyenphamanh1996',
                'huynguyenhuynh1997',
                'trinhlekhac1992',
                'quangminhvu2000',
                'nguyenanhphamnguyen1996',
                'huyhuynhnguyen1997',
                'trinhkhacle1992',
            ]
        })

        extracted_name = name_extractor.extract_username(
            email_name_data,
            email_name_col='email_name'
        )['username_extracted'].tolist()
        pprint(extracted_name, compact=True)

        assert extracted_name == [
            'Quang Vu Minh',
            'Nguyen Nguyen Pham Anh',
            'Huy Nguyen Huynh',
            'Trinh Le Khac',
            'Quang Minh Vu',
            'Nguyen Anh Pham Nguyen',
            'Huy Huynh Nguyen',
            'Trinh Khac Le',
        ]

    @pt.mark.email2name
    def test_extract_first_name_basic(self):
        """
        Test whether the function can extract for firstname name in email name
        with basic name
        """
        name_extractor = EmailNameExtractor()
        email_name_data = pd.DataFrame.from_dict({
            'email_name': [
                'quang2000',
                'nguyen1996',
                'huy1997',
                'trinh1992',
            ]
        })

        extracted_name = name_extractor.extract_username(
            email_name_data,
            email_name_col='email_name'
        )['username_extracted'].tolist()
        pprint(extracted_name, compact=True)

        assert extracted_name == [
            'Quang',
            'Nguyen',
            'Huy',
            'Trinh',
        ]

    # * Testing for YOB
    @pt.mark.email2yob
    def test_extract_full_year_only_match(self):
        """
        Test whether the function can extract for full year in name string
        that have correct yob
        """
        yob_extractor = EmailYOBExtractor()

        email_name_data = pd.DataFrame.from_dict({
            'email_name': [
                'caspernguyen1980',
                'cunanh1999',
                'quangvm2000',
                'quangvm2009',
            ]
        })

        extracted_yob = yob_extractor.extract_yob(
            email_name_data,
            email_name_col='email_name'
        )['yob_extracted'].tolist()

        assert extracted_yob == [1980.0, 1999.0, 2000.0, 2009.0]

    @pt.mark.email2yob
    def test_extract_full_year_only_not_match(self):
        """
        Test whether the function can extract for full year in name string
        that not match correct yob
        """
        yob_extractor = EmailYOBExtractor()

        email_name_data = pd.DataFrame.from_dict({
            'email_name': [
                'minhquan2012',
                'minhvu1949',
            ]
        })

        extracted_yob = yob_extractor.extract_yob(
            email_name_data,
            email_name_col='email_name'
        )['yob_extracted'].tolist()

        assert all(math.isnan(yob) for yob in extracted_yob)

    @pt.mark.email2yob
    def test_extract_half_year_only_match(self):
        """
        Test whether the function can extract for half year in name string
        that have correct yob
        """
        yob_extractor = EmailYOBExtractor()

        email_name_data = pd.DataFrame.from_dict({
            'email_name': [
                'henryvu09',
                'quangvu00',
                'quangvu20',
                'minhnhat50'
            ]
        })

        extracted_yob = yob_extractor.extract_yob(
            email_name_data,
            email_name_col='email_name'
        )['yob_extracted'].tolist()

        assert extracted_yob == [2009.0, 2000.0, 2000.0, 1950.0]

    @pt.mark.email2yob
    def test_extract_half_year_only_not_match(self):
        """
        Test whether the function can extract for half year in name string
        that not match correct yob
        """
        yob_extractor = EmailYOBExtractor()

        email_name_data = pd.DataFrame.from_dict({
            'email_name': [
                'minhvu19',
                'henry49',
            ]
        })

        extracted_yob = yob_extractor.extract_yob(
            email_name_data,
            email_name_col='email_name'
        )['yob_extracted'].tolist()

        assert all(math.isnan(yob) for yob in extracted_yob)

    # * Testing for phone
    @pt.mark.email2phone
    def test_extract_mobi_phone_number(self):
        """
        Test whether the function can extract for mobi phone from name string
        """
        phone_extractor = EmailPhoneExtractor()

        email_name_data = pd.DataFrame.from_dict({
            'email_name': [
                '0889845985quangvm',
                '0937044781anhem',
            ]
        })

        extracted_phone = phone_extractor.extract_phone(
            email_name_data,
            email_name_col='email_name'
        )['phone_extracted'].tolist()

        assert extracted_phone == ['0889845985', '0937044781']

    @pt.mark.email2phone
    def test_extract_landline_phone_number(self):
        """
        Test whether the function can extract for landline phone from name string
        """
        phone_extractor = EmailPhoneExtractor()

        email_name_data = pd.DataFrame.from_dict({
            'email_name': [
                '02293187374quangvm',
                '02258614680anhem',
            ]
        })

        extracted_phone = phone_extractor.extract_phone(
            email_name_data,
            email_name_col='email_name'
        )['phone_extracted'].tolist()

        assert extracted_phone == ['02293187374', '02258614680']

    @pt.mark.email2phone
    def test_extract_mobi_phone_number_from_complex_name(self):
        """
        Test whether the function can extract for mobi phone from name string
        with both mobi phone and card-id
        """
        phone_extractor = EmailPhoneExtractor()

        email_name_data = pd.DataFrame.from_dict({
            'email_name': [
                '0889845985-077200004509quangvm',
                '0937044781-077200004509quangvm',
            ]
        })

        extracted_phone = phone_extractor.extract_phone(
            email_name_data,
            email_name_col='email_name'
        )['phone_extracted'].tolist()

        assert extracted_phone == ['0889845985', '0937044781']

    # * Testing for address
    @pt.mark.email2address
    def test_extract_simple_address_v1(self):
        """
        Test whether the function can extract for address from name string -- Thành Phố
        """
        address_extractor = EmailAddressExtractor()

        email_name_data = pd.DataFrame.from_dict({
            'email_name': [
                'hanoi',
                'hochiminh',
            ]
        })

        extracted_address = address_extractor.extract_address(
            email_name_data,
            email_name_col='email_name'
        )['address_extracted'].tolist()

        assert extracted_address == [
            'Thành phố Hà Nội',
            'Thành phố Hồ Chí Minh'
        ]

    @pt.mark.email2address
    def test_extract_simple_address_v2(self):
        """
        Test whether the function can extract for address from name string -- Tỉnh
        """
        address_extractor = EmailAddressExtractor()

        email_name_data = pd.DataFrame.from_dict({
            'email_name': [
                'bariavungtau',
                'camau',
                'hungyen',
                'dongnai',
                'longan'
            ]
        })

        extracted_address = address_extractor.extract_address(
            email_name_data,
            email_name_col='email_name'
        )['address_extracted'].tolist()

        assert extracted_address == [
            'Tỉnh Bà Rịa - Vũng Tàu',
            'Tỉnh Cà Mau',
            'Tỉnh Hưng Yên',
            'Tỉnh Đồng Nai',
            'Tỉnh Long An'
        ]

    @pt.mark.email_valid
    def test_validate_large_company_email(self):
        """
        Test whether the function can validate correctly for company email
        """
        email_data = pd.DataFrame.from_dict({
            'email': [
                'quangvm2000@gmail.com',
                'vuminhquang2000@gmail.com',
                '0889845985@gmail.com',
                'quang0889845985@gmail.com'
            ]
        })

        validated_email = process_validate_email(
            email_data,
            email_col='email'
        )['is_email_valid'].tolist()

        assert validated_email == [
            True,
            True,
            False,
            True
        ]

    @pt.mark.email_valid
    def test_validate_edu_email(self):
        """
        Test whether the function can validate correctly for edu email
        """
        email_data = pd.DataFrame.from_dict({
            'email': [
                'quang.vuminh@hcmut.edu.vn',
                '1852699@hcmut.edu.vn',
                '6699@hcmut.edu.vn',
            ]
        })

        validated_email = process_validate_email(
            email_data,
            email_col='email'
        )['is_email_valid'].tolist()

        assert validated_email == [
            True,
            True,
            False,
        ]
