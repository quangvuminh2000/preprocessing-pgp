"""
Constants for processing name type extraction
"""

import os
import json
from configparser import ConfigParser
from flashtext import KeywordProcessor

# ? NAME TYPE RULE
NAME_TYPE_PATH = os.path.join(
    os.path.dirname(__file__),
    '../../',
    'data',
    'name_type',
    'customer_type.ini'
)
NAME_TYPE_CONFIG = ConfigParser()
NAME_TYPE_CONFIG.read(NAME_TYPE_PATH)

# ? REGEX
EXCLUDE_REGEX = {
    'company': 'benh vien|ngan hang',
    'biz': None,
    'edu': None,
    'medical': None
}

TYPE_NAMES = {
    'company': 'Cong ty',
    'biz': 'Ho kinh doanh',
    'edu': 'Giao duc',
    'medical': 'Benh vien - Phong kham'
}

# ? CUSTOMER TYPE KEYWORDS -- lv1, lv2
KEYWORDS = {}
for name_type in NAME_TYPE_CONFIG.sections():
    KEYWORDS[name_type] = {}
    type_sections = NAME_TYPE_CONFIG[name_type]
    for level in ['lv1', 'lv2']:
        type_lvl_kws = KeywordProcessor(case_sensitive=True)

        kws_list = json.load(type_sections[level])
        type_lvl_kws.add_keywords_from_list(kws_list)

        KEYWORDS[name_type][level] = type_lvl_kws
