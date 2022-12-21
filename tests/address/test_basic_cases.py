"""
Testing for basic cases of address extractions
"""
# import os
# import pandas as pd

# import pytest

from preprocessing_pgp.address import (
    level_extractor,
    preprocess
)


class TestLevelExtraction:
    """
    Class for testing for address level name extraction functionalities
    """

    extractor = level_extractor.LevelExtractor()
    cleaner = preprocess.VietnameseAddressCleaner()

    # * Testing for level 1
    def test_enriched_lv1_extraction(self):
        """
        Test it can extract level 1 correctly from a string of address

        * Level 1 is enriched and clean
        * The address not containing any abbreviation
        """

        address_data = '''658/11 Trương Công Định,
        Phường Nguyễn An Ninh,
        Thành Phố Vũng Tàu, Tỉnh Bà Rịa - Vũng Tàu'''

        cleaned_addr = self.cleaner.clean_address(address_data)

        _, _, best_lvs = self.extractor.extract_all_levels(
            cleaned_addr)

        assert best_lvs[1] == 'Tỉnh Bà Rịa - Vũng Tàu'

    def test_non_accented_lv1_extraction(self):
        """
        Test it can extract level 1 correctly from a string of address

        * Level 1 is non-accented and clean
        * The address not containing any abbreviation
        """

        address_data = '''658/11 Truong Cong Dinh,
        Phuong Nguyen An Ninh,
        Thanh Pho Vung Tau, Tinh Ba Ria - Vung Tau'''

        cleaned_addr = self.cleaner.clean_address(address_data)

        _, _, best_lvs = self.extractor.extract_all_levels(
            cleaned_addr)

        assert best_lvs[1] == 'Tỉnh Bà Rịa - Vũng Tàu'

    def test_enriched_lv2_extraction(self):
        """
        Test it can extract level 2 correctly from a string of address

        * Level 2 is enriched and clean
        * The address not containing any abbreviation
        """

        address_data = '''658/11 Trương Công Định,
        Phường Nguyễn An Ninh,
        Thành Phố Vũng Tàu, Tỉnh Bà Rịa - Vũng Tàu'''

        cleaned_addr = self.cleaner.clean_address(address_data)

        _, _, best_lvs = self.extractor.extract_all_levels(
            cleaned_addr)

        assert best_lvs[2] == 'Thành phố Vũng Tàu'

    def test_non_accented_lv2_extraction(self):
        """
        Test it can extract level 2 correctly from a string of address

        * Level 2 is non-accented and clean
        * The address not containing any abbreviation
        """

        address_data = '''658/11 Truong Cong Dinh,
        Phuong Nguyen An Ninh,
        Thanh Pho Vung Tau, Tinh Ba Ria - Vung Tau'''

        cleaned_addr = self.cleaner.clean_address(address_data)

        _, _, best_lvs = self.extractor.extract_all_levels(
            cleaned_addr)

        assert best_lvs[2] == 'Thành phố Vũng Tàu'

    def test_enriched_lv3_extraction(self):
        """
        Test it can extract level 3 correctly from a string of address

        * Level 3 is enriched and clean
        * The address not containing any abbreviation
        """

        address_data = '''658/11 Trương Công Định,
        Phường Nguyễn An Ninh,
        Thành Phố Vũng Tàu, Tỉnh Bà Rịa - Vũng Tàu'''

        cleaned_addr = self.cleaner.clean_address(address_data)

        _, _, best_lvs = self.extractor.extract_all_levels(
            cleaned_addr)

        assert best_lvs[3] == 'Phường Nguyễn An Ninh'

    def test_non_accented_lv3_extraction(self):
        """
        Test it can extract level 3 correctly from a string of address

        * Level 3 is non-accented and clean
        * The address not containing any abbreviation
        """

        address_data = '''658/11 Truong Cong Dinh,
        Phuong Nguyen An Ninh,
        Thanh Pho Vung Tau, Tinh Ba Ria - Vung Tau'''

        cleaned_addr = self.cleaner.clean_address(address_data)

        _, _, best_lvs = self.extractor.extract_all_levels(
            cleaned_addr)

        assert best_lvs[3] == 'Phường Nguyễn An Ninh'

    def test_enriched_remained_addr_extraction(self):
        """
        Test it can extract remaining address nearly correct from a string of address

        * Remaining address is the remained string after extraction
        * The address not containing any abbreviation
        """

        address_data = '''658/11 Trương Công Định,
        Phường Nguyễn An Ninh,
        Thành Phố Vũng Tàu, Tỉnh Bà Rịa - Vũng Tàu'''

        cleaned_addr = self.cleaner.clean_address(address_data)

        _, remained_address, _ = self.extractor.extract_all_levels(
            cleaned_addr)

        # assert remained_address.find('658/11 truong cong dinh') != -1
        assert remained_address == '658/11 truong cong dinh'

    def test_non_accented_remained_addr_extraction(self):
        """
        Test it can extract remaining address nearly correct from a string of address

        * Remaining address is the remained string after extraction
        * The address not containing any abbreviation
        """

        address_data = '''658/11 Truong Cong Dinh,
        Phuong Nguyen An Ninh,
        Thanh Pho Vung Tau, Tinh Ba Ria - Vung Tau'''

        cleaned_addr = self.cleaner.clean_address(address_data)

        _, remained_address, _ = self.extractor.extract_all_levels(
            cleaned_addr)

        assert remained_address.find('658/11 truong cong dinh') != -1
