"""
File containing const that is necessary for processing address extraction
"""

from itertools import product
from abc import ABC
from dataclasses import dataclass
from typing import List


# ? REGEX FOR ADDRESS
DICT_NORM_ABBREV_REGEX_KW = {
    'tp ': [r'\btp\.\b', r'\btp:\b'],
    'tt ': [r'\btt\.\b', r'\btt:\b'],
    'q ': [r'\bq\.\b', r'\bq:\b'],
    'h ': [r'\bh\.\b', r'\bh:\b'],
    'x ': [r'\bx\.\b', r'\bx:\b'],
    'p ': [r'\bp\.\b', r'\bp:\b']
}

DICT_NORM_CITY_DASH_REGEX = {
    'ba ria - vung tau': ['\bba ria vung tau\b'],
    'br-vt': ['\bbrvt\b'],
    'phan rang - thap cham': ['\bphan rang thap cham\b'],
    'pr-tc': ['\bprtc\b']
}

# ? LEVEL METHODS
@dataclass
class ExtractLevelMethod(ABC):
    LV1_METHODS = ['lv1_norm', 'lv1_abbrev', 'lv1_prefix_im', 'lv1_nprefix_im']
    LV2_METHODS = ['lv2_norm', 'lv2_abbrev', 'lv2_prefix_im', 'lv2_nprefix_im']
    LV3_METHODS = ['lv3_norm', 'lv3_abbrev', 'lv3_prefix_im', 'lv3_nprefix_im']

    @staticmethod
    def get_combined_level_methods(n_level: int=1) -> List:
        """
        Function to get combined level methods based on the number of levels to combined

        Parameters
        ----------
        n_level : int, optional
            The number of level (1-3) to combined, by default 1

        Returns
        -------
            The output list of dictionaries for each of the levels
        """
