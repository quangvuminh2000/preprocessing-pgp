import os

N_PROCESSES = os.cpu_count() // 2

DICT_TRASH_STRING = {
    '': None,
    'Nan': None,
    'nan': None,
    'None': None,
    'none': None,
    'Null': None,
    'null': None,
    '""': None
}
