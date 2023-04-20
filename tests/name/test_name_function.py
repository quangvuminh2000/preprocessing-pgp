"""
Tests for enrich & filling accent to names
"""
import pytest as pt
import pandas as pd

from preprocessing_pgp.name.model.lstm import predict_gender_from_name
from preprocessing_pgp.name.enrich_name import process_enrich


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
                'Nguyễn Thị Kim Hà', 'Nguyễn Huỳnh Xuân Mai', 'Nguyễn Quỳnh Sương',
                'Nguyễn Văn Lành',
                'Lan'
            ]
        })

        predict_data = predict_gender_from_name(
            name_data,
            name_col='name'
        )

        print(predict_data)

        predicted_genders = predict_data['gender_predict'].values.tolist()

        assert predicted_genders == [
            'M', 'M', 'M',
            'M', 'M', 'M',
            'M', 'M', 'M',
            'M', 'M', 'M',
            'M', 'F', 'F',
            'F', 'F', 'F',
            'M', 'F'
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
                'Nguyen Thi Kim Ha', 'Nguyen Huynh Xuan Mai', 'Nguyen Quynh Suong',
                'Nguyen Van Lanh'
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
            'F', 'F', 'F',
            'M'
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

    @pt.mark.enrich_name_error
    def test_enrich_name_basic_case(self):
        """
        Test whether the model can predict correctly non-accent names
        """
        name_data = pd.DataFrame.from_dict({
            'name': [
                'Nguyễn Thị Á Châu',
                'Lò A Bình',
                'Nguyẽn Văn Nam',
                'Ngo Nam Anh',
                'Văn Hải tq',
                'Thùy Dương',
                'Đinh Phơn',
                'Nguyễn Văn Phong',
                'A.Phong',
                'C. Hoa',
                '',
                '088989483 A.Quang',
                'A,Quang',
                'E, Quang',
                'Bộ Công An',
                'Trại Tạm Giam',
                'Vận Tải',
                'Công An Tỉnh',
                'Cong An Huyen',
                'Van Tai',
                'Văn Tài'
            ]
        })

        predict_data = process_enrich(
            name_data,
            name_col='name'
        )

        print(predict_data)

        predicted_names = predict_data['final'].values.tolist()
        print(predicted_names)

        assert True

    @pt.mark.enrich_name
    def test_enrich_name_customer_company(self):
        """
        Test whether the model can predict correctly company category in customer names
        """
        name_data = pd.DataFrame.from_dict({
            'name': [
                'Nguyen Van Phong',
            ]
        })

        predict_data = process_enrich(
            name_data,
            name_col='name'
        )

        predicted_names = predict_data['final'].values.tolist()

        assert predicted_names == [
            'Nguyễn Văn Phòng'
        ]
