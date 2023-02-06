"""
Module for essential constants in enrich name
"""
import os

import pandas as pd

# ? MODEL PATHS
NAME_SPLIT_PATH = os.path.join(
    os.path.dirname(__file__),
    os.pardir,
    'data',
    'name_split'
)

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    os.pardir,
    'data',
    'transformer_model',
    'trial-25'
)

RULE_BASED_PATH = os.path.join(
    os.path.dirname(__file__),
    os.pardir,
    'data',
    'rule_base',
    'students'
)

NICKNAME_PATH = os.path.join(
    os.path.dirname(__file__),
    '../data/name_split/nicknames.parquet'
)

# ? PREPROCESS CONSTANTS
NON_HUMAN_REG_LIST = [
    # Companies
    'Cong Ty', 'Co Phan', 'Co Phieu', 'Bds', 'Bat Dong San', 'Tnhh', 'Thuong Nghiep', 'Huu Hang',
    # Schools
    # 'Doan Truong', 'Van Phong',
    'Truong Hoc', 'Thpt', 'Thcs', 'Dai Hoc', 'Tieu Hoc', 'Noi Tru', 'Ngoai Tru', 'Cao Hoc', 'Thac Si', 'Tien Si', 'Phd', 'Pho Giao Su', 'Hoc Vien',
    # Have Meanings
    'To Phuong', 'Nghi Viec', 'Thoi Viec', 'Lam Viec', 'So Y Te', 'Ve Sinh',
    # Others
    'Ca Bong', 'Ca Chep'
]

'''
NON-OFFICIAL NAMES : Ref from 'JOURNAL OF ETHNIC MINORITIES RESEARCH - 2019'

- NON_OFFICIAL_NAMES: Tên không chính thức có dấu của các dân tộc ở Việt Nam.
- LOCAL_GROUP_NAMES: Tên chỉ các nhóm địa phương, nhóm riêng.
'''
NON_OFFICIAL_NAMES = [
    # Kinh
    'Keo', 'Doan',
    # Khmer
    'Cur', 'Cul', 'Khơ Me', 'Krôm',
    # Mường
    'Mol', 'Mual', 'Moi'
    # Mông
    'Mèo', 'Mẹo'
    # Dao
    'Mán', 'Động', 'Trại', 'Dìu Miền', 'Kiềm Miền',
    # Ngái
    'Xín', 'Lê', 'Đán', 'Khách Gia',
    # Sán Chay
    'Hờn Bạn', 'Sơn Tử',
    # Cơ Ho
    'Còn Chau',
    # Chăm
    'Chàm', 'Chiêm',
    # Sán Dìu
    'Trại', 'Trại Đất', 'Mán Quần Cộc',
    # Hrê
    'Chăm Hrê', 'Chom, Lũy',
    # Mnông
    'Pnông', 'Nông',
    # Giáy
    'Nhắng', 'Dẩng',
    # Gié - Triêng
    'Giang Rẫy',
    # Mạ
    'Còn Chau', 'Chau Mạ',
    # Khơ Mú
    'Xá', 'Xá Cẩu', 'Mứn Xen', 'Pu Thênh', 'Tềnh', 'Tày Hạy'
]

LOCAL_GROUP_NAMES = [
    # Tày
    'Tày', 'Ngạn', 'Pa Dí', 'Thu Lao',
    # Gia Rai
    'Chor', 'Hđrung', 'Arap', 'Mthur', 'Tbuăn',
    # Chứt
    'Arem', 'Rục', 'Mày', 'Sách', 'Mã Liềng', 'Kri',
    # Ê Đê
    'Kpă', 'Mthur', 'Ktul', 'Đliê', 'Hruê', 'Blô', 'Ê Pan', 'Bih', 'Krung', 'Kđrao',
    # Thổ
    'Thổ', 'Kẹo', 'Họ', 'Mọn', 'Cuối', 'Tày Poọng', 'Đan Lai',
    # Xơ Đăng
    'Xơ Teng', 'Hđang', 'Tơ Đrá', 'Mơ Nâm', 'Ha Lăng', 'Ca Dong',
    # Bru - Vân Kiều
    'Khùa', 'Mang Coong', 'Trì', 'Sộ', 'Vân Kiều',
    # Giẻ - Triêng
    'Giẻ', 'Triêng', 'Ta Liêng', 'Ve', 'Bơ Noong', 'Pơ Noong'
]


REPLACE_HUMAN_REG_DICT = {
    'K HI': '',
    'Bs': '',
    'Ng.': 'Nguyễn'
}

BRIEF_NAME_DICT = {
    'nguyen': ['ng.', 'ng', 'n.', 'n'],
    'do': ['d.', 'd'],
    'pham': ['p.', 'p']
}

# * NICKNAMES
NICKNAMES = pd.read_parquet(NICKNAME_PATH)
NICKNAME_REGEX = '|'.join(
    [*NICKNAMES['name'].to_list(),
     *NICKNAMES[NICKNAMES['de_name'].str.split().str.len() > 1]['de_name'].to_list()]
)
