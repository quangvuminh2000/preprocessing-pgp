"""
Tests for preprocessing names
"""

import pytest as pt
import pandas as pd

class TestPreprocessName:
    """
    Class contains tests for name preprocessing issues
    """

    @pt.mark.preprocess_name
    def test_predict_gender_accented_name(self):
        """
        Test whether the model can predict correctly accented names
        """
        name_data = pd.DataFrame.from_dict({
            'name': [
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