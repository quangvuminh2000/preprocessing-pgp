"""
Tests for card id validator
"""

from preprocessing_pgp.card import (
    validation
)


class TestPersonalIDValidator:
    """
    Class for testing for Personal Id card validator mechanism
    """

    # * All of Genders
    def test_valid_gender_male_before_cen_20(self):
        """
        Male: 0
        YOB: 1990
        """

        card_id = '079090002002'

        is_valid = validation.PersonalIDValidator.is_valid_card(card_id)

        assert is_valid

    def test_valid_gender_old_male_before_cen_20(self):
        """
        Male: 0
        YOB: 1908
        """

        card_id = '077008002002'

        is_valid = validation.PersonalIDValidator.is_valid_card(card_id)

        assert is_valid

    def test_valid_gender_male_after_cen_20(self):
        """
        Male: 2
        YOB: 2000
        """

        card_id = '077200002002'

        is_valid = validation.PersonalIDValidator.is_valid_card(card_id)

        assert is_valid

    def test_invalid_gender_male_after_cen_20_case_1(self):
        """
        Male: 2
        YOB: 2009 --> Not enough old limit for making personal id card
        """
        card_id = '0122096789012'

        is_valid = validation.PersonalIDValidator.is_valid_card(card_id)

        assert ~is_valid

    def test_invalid_gender_male_after_cen_20_case_2(self):
        """
        Male: 2
        YOB: 2045 --> Not even born yet
        """
        card_id = '0122456789012'

        is_valid = validation.PersonalIDValidator.is_valid_card(card_id)

        assert ~is_valid

    def test_valid_gender_female_before_cen_20(self):
        """
        Female: 1
        YOB: 1995
        """
        card_id = '077195008579'

        is_valid = validation.PersonalIDValidator.is_valid_card(card_id)

        assert is_valid

    def test_valid_gender_old_female_before_cen_20(self):
        """
        Female: 1
        YOB: 1905
        """
        card_id = '077105008579'

        is_valid = validation.PersonalIDValidator.is_valid_card(card_id)

        assert is_valid

    def test_valid_gender_female_after_cen_20(self):
        """
        Female: 3
        YOB: 2005
        """
        card_id = '077305008579'

        is_valid = validation.PersonalIDValidator.is_valid_card(card_id)

        assert is_valid

    def test_invalid_gender_female_after_cen_20_case_1(self):
        """
        Female: 3
        YOB: 2015 -> not old enough for personal id card
        """
        card_id = '077315009830'

        is_valid = validation.PersonalIDValidator.is_valid_card(card_id)

        assert ~is_valid

    def test_invalid_gender_female_after_cen_20_case_2(self):
        """
        Female: 3
        YOB: 2035 --> Not even born yet
        """
        card_id = '077335009830'

        is_valid = validation.PersonalIDValidator.is_valid_card(card_id)

        assert ~is_valid

    def test_invalid_region_code_case_1(self):
        """
        Region code: 003 --> Not have yet
        """
        card_id = '003305009830'

        is_valid = validation.PersonalIDValidator.is_valid_card(card_id)

        assert ~is_valid

    def test_invalid_old_code_case_1(self):
        """
        Region code: 28 --> Not have yet
        """
        card_id = '283704895'

        is_valid = validation.PersonalIDValidator.is_valid_card(card_id)

        assert ~is_valid

    def test_valid_old_code_case_1(self):
        """
        Region code: 27
        """
        card_id = '273704895'

        is_valid = validation.PersonalIDValidator.is_valid_card(card_id)

        assert is_valid
