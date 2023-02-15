"""
Tests for email info extraction
"""
import pytest as pt
import pandas as pd

from preprocessing_pgp.name.model.lstm import predict_gender_from_name

class TestNameFunction:
    """
    Class contains tests for information extraction from email
    """

    # * Testing for name2gender
    @pt.mark.name2gender
    def test_predict_gender_accented_name(self):
        """
        Test whether the model can predict correctly accented names
        """
        name_data = pd.DataFrame.from_dict({
            'name': [
                'Vũ Minh Quang', 'Nguyễn Huỳnh Huy', 'Nguyễn Phạm Anh Nguyên',
                'Lê Khắc Trình', 'Đặng Phương Nam', 'Hùng Ngọc Phát',
                'Ngô Triệu Long', 'Lê Duy Long', 'Hoàng Bảo Khánh',
                'Nguyễn Khắc Toàn', 'Trương Quang Hoàng', 'Phạm Minh Quang',
                'Ngô Hoàng Khôi', 'Võ Ngọc Trăm', 'Trần Ngọc Lan Khanh',
                'Nguyễn Thị Kim Hà', 'Nguyễn Huỳnh Xuân Mai', 'Nguyễn Quỳnh Sương'
            ]
        })

        predicted_genders = predict_gender_from_name(
            name_data,
            name_col='name'
        )['gender_predict'].values.tolist()

        assert predicted_genders == [
            'M', 'M', 'M',
            'M', 'M', 'M',
            'M', 'M', 'M',
            'M', 'M', 'M',
            'M', 'F', 'F',
            'F', 'F', 'F'
        ]

    @pt.mark.name2gender
    def test_predict_gender_non_accented_name(self):
        """
        Test whether the model can predict correctly accented names
        """
        name_data = pd.DataFrame.from_dict({
            'name': [
                'Vu Minh Quang', 'Nguyen Huynh Huy', 'Nguyen Pham Anh Nguyen',
                'Le Khac Trinh', 'Dang Phuong Nam', 'Hung Ngoc Phat',
                'Ngo Trieu Long', 'Le Duy Long', 'Hoang Bao Khanh',
                'Nguyen Khac Toan', 'Truong Quang Hoang', 'Pham Minh Quang',
                'Ngo Hoang Khoi', 'Vo Ngoc Tram', 'Tran Ngoc Lan Khanh',
                'Nguyen Thi Kim Ha', 'Nguyen Huynh Xuan Mai', 'Nguyen Quynh Suong'
            ]
        })

        predicted_genders = predict_gender_from_name(
            name_data,
            name_col='name'
        )['gender_predict'].values.tolist()

        assert predicted_genders == [
            'M', 'M', 'M',
            'M', 'M', 'M',
            'M', 'M', 'M',
            'M', 'M', 'M',
            'M', 'F', 'F',
            'F', 'F', 'F'
        ]

    @pt.mark.name2gender
    def test_predict_gender_edge_accented_name(self):
        """
        Test whether the model can predict correctly edge accented names
        """
        name_data = pd.DataFrame.from_dict({
            'name': [
                'Nguyễn Hữu Đức'
            ]
        })

        predicted_genders = predict_gender_from_name(
            name_data,
            name_col='name'
        )['gender_predict'].values.tolist()

        assert predicted_genders == [
            'M'
        ]
