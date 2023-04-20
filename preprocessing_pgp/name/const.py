"""
Module for essential constants in enrich name
"""
import os

import pandas as pd

# ? MODEL PATHS
NAME_SPLIT_PATH = os.path.join(
    os.path.dirname(__file__), os.pardir, "data", "name_split"
)

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    os.pardir,
    "data",
    "transformer_model",
    "trial-31",
)

# ? RULE-BASED PATH
RULE_BASED_PATH = os.path.join(
    os.path.dirname(__file__), os.pardir, "data", "rule_base", "students"
)

NICKNAME_PATH = os.path.join(
    os.path.dirname(__file__), "../data/name_split/nicknames_boundary.parquet"
)

GENDER_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "../data/gender_model"
)

GENDER_MODEL_VERSION = "1.0"

PRONOUN_GENDER_RB_PATH = os.path.join(
    os.path.dirname(__file__),
    "../data/gender_model/rule_base/pronoun_gender_dict.parquet",
)
PRONOUN_GENDER_DF = pd.read_parquet(PRONOUN_GENDER_RB_PATH)
PRONOUN_GENDER_MAP = dict(
    zip(PRONOUN_GENDER_DF["pronoun"], PRONOUN_GENDER_DF["gender"])
)

NAME_ELEMENT_PATH = os.path.join(
    os.path.dirname(__file__), "../data/name_split/name_elements.parquet"
)

MF_NAME_GENDER_RULE = pd.read_parquet(
    os.path.join(
        os.path.dirname(__file__), "../data/rule_base/gender/mfname.parquet"
    )
)

BEFORE_FNAME_GENDER_RULE = pd.read_parquet(
    os.path.join(
        os.path.dirname(__file__),
        "../data/rule_base/gender/before_fname.parquet",
    )
)

# ? PREPROCESS CONSTANTS
NON_HUMAN_REG_LIST = [
    # Companies
    "Cong Ty",
    "Co Phan",
    "Co Phieu",
    "Bds",
    "Bat Dong San",
    "Tnhh",
    "Thuong Nghiep",
    "Huu Hang",
    # Schools
    # 'Doan Truong', 'Van Phong',
    "Truong Hoc",
    "Thpt",
    "Thcs",
    "Dai Hoc",
    "Tieu Hoc",
    "Noi Tru",
    "Ngoai Tru",
    "Cao Hoc",
    "Thac Si",
    "Tien Si",
    "Phd",
    "Pho Giao Su",
    "Hoc Vien",
    # Have Meanings
    "To Phuong",
    "Nghi Viec",
    "Thoi Viec",
    "Lam Viec",
    "So Y Te",
    "Ve Sinh",
    # Others
    "Ca Bong",
    "Ca Chep",
]

PRONOUN_REGEX = r"^(?:\bkh\b|\bkhach hang\b|\bchị\b|\bchi\b|\banh\b|\ba\b|\bchij\b|\bc\b|\be\b|\bem\b|\bcô\b|\bco\b|\bchú\b|\bbác\b|\bbac\b|\bme\b|\bdì\b|\bông\b|\bong\b|\bbà\b)\s+"
PRONOUN_REGEX_W_DOT = r"^(?:\bkh\b|\bkhach hang\b|\bchị\b|\bchi\b|\banh\b|\ba\b|\bchij\b|\bc\b|\be\b|\bem\b|\bcô\b|\bco\b|\bchú\b|\bbác\b|\bbac\b|\bme\b|\bdì\b|\bông\b|\bong\b|\bbà\b|\ba|\bc)[.,]"

"""
NON-OFFICIAL NAMES : Ref from 'JOURNAL OF ETHNIC MINORITIES RESEARCH - 2019'

- NON_OFFICIAL_NAMES: Tên không chính thức có dấu của các dân tộc ở Việt Nam.
- LOCAL_GROUP_NAMES: Tên chỉ các nhóm địa phương, nhóm riêng.
"""
NON_OFFICIAL_NAMES = [
    # Kinh
    "Keo",
    "Doan",
    # Khmer
    "Cur",
    "Cul",
    "Khơ Me",
    "Krôm",
    # Mường
    "Mol",
    "Mual",
    "Moi"
    # Mông
    "Mèo",
    "Mẹo"
    # Dao
    "Mán",
    "Động",
    "Trại",
    "Dìu Miền",
    "Kiềm Miền",
    # Ngái
    "Xín",
    "Lê",
    "Đán",
    "Khách Gia",
    # Sán Chay
    "Hờn Bạn",
    "Sơn Tử",
    # Cơ Ho
    "Còn Chau",
    # Chăm
    "Chàm",
    "Chiêm",
    # Sán Dìu
    "Trại",
    "Trại Đất",
    "Mán Quần Cộc",
    # Hrê
    "Chăm Hrê",
    "Chom, Lũy",
    # Mnông
    "Pnông",
    "Nông",
    # Giáy
    "Nhắng",
    "Dẩng",
    # Gié - Triêng
    "Giang Rẫy",
    # Mạ
    "Còn Chau",
    "Chau Mạ",
    # Khơ Mú
    "Xá",
    "Xá Cẩu",
    "Mứn Xen",
    "Pu Thênh",
    "Tềnh",
    "Tày Hạy",
]

LOCAL_GROUP_NAMES = [
    # Tày
    "Tày",
    "Ngạn",
    "Pa Dí",
    "Thu Lao",
    # Gia Rai
    "Chor",
    "Hđrung",
    "Arap",
    "Mthur",
    "Tbuăn",
    # Chứt
    "Arem",
    "Rục",
    "Mày",
    "Sách",
    "Mã Liềng",
    "Kri",
    # Ê Đê
    "Kpă",
    "Mthur",
    "Ktul",
    "Đliê",
    "Hruê",
    "Blô",
    "Ê Pan",
    "Bih",
    "Krung",
    "Kđrao",
    # Thổ
    "Thổ",
    "Kẹo",
    "Họ",
    "Mọn",
    "Cuối",
    "Tày Poọng",
    "Đan Lai",
    # Xơ Đăng
    "Xơ Teng",
    "Hđang",
    "Tơ Đrá",
    "Mơ Nâm",
    "Ha Lăng",
    "Ca Dong",
    # Bru - Vân Kiều
    "Khùa",
    "Mang Coong",
    "Trì",
    "Sộ",
    "Vân Kiều",
    # Giẻ - Triêng
    "Giẻ",
    "Triêng",
    "Ta Liêng",
    "Ve",
    "Bơ Noong",
    "Pơ Noong",
]


REPLACE_HUMAN_REG_DICT = {"K HI": "", "Bs": "", "Ng.": "Nguyễn"}

BRIEF_NAME_DICT = {"nguyen": ["ng.", "n."], "do": ["d."], "pham": ["p."]}

# * NICKNAMES
NICKNAMES = pd.read_parquet(NICKNAME_PATH)
NICKNAME_REGEX = "|".join(
    [
        *NICKNAMES["name"].to_list(),
        *NICKNAMES[NICKNAMES["de_name"].str.split().str.len() > 1][
            "de_name"
        ].to_list(),
    ]
)

# * NAME POSSIBLE ELEMENTS
NAME_ELEMENTS = pd.read_parquet(NAME_ELEMENT_PATH)
WITHOUT_ACCENT_ELEMENTS = set(NAME_ELEMENTS["without_accent"].unique())
WITH_ACCENT_ELEMENTS = set(NAME_ELEMENTS["with_accent"].unique())
