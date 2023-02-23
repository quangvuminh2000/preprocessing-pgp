"""
Module contains constants for email processing
"""

# ? NONE TYPE EMAIL DICT
NAN_EMAIL_LIST = ['nan', 'none', '']

# ? EMAIL REGEX

AT_LEAST_ONE_CHAR_REGEX = r'(?=.*[a-z])'

EMAIL_DOMAIN_REGEX = {
    'gmail': {
        'domains': ['gmail.com', 'gmail.com.vn'],
        'regex': r'^[a-z0-9][a-z0-9\.]{8,28}[a-z0-9]@[a-z0-9]{2,}(?:\.[a-z0-9]{2,12}){1,2}$'
    },
    'yahoo': {
        'domains': ['yahoo.com', 'yahoo.com.vn'],
        'regex': r'^[a-z][a-z0-9_\.]{2,30}[a-z0-9]@[a-z0-9]{2,}(?:\.[a-z0-9]{2,12}){1,2}$'
    },
    'ms': {
        'domains': ['hotmail.com', 'outlook.com', 'outlook.com.vn'],
        'regex': r'^[a-z][a-z0-9-_\.]{2,62}[a-z0-9-_]@[a-z0-9]{2,}(?:\.[a-z0-9]{2,12}){1,2}$'
    },
    'fpt': {
        'domains': ['fpt.com.vn', 'fpt.edu.vn',
                    'hcm.fpt.vn', 'fpt.vn',
                    'fpt.net', 'fpt.aptech.ac.vn'],
        'regex': r'^[a-z0-9][a-z0-9_\.]{2,31}[a-z0-9]@[a-z0-9]{2,}(?:\.[a-z0-9]{2,12}){1,2}$'
    }
}

COMMON_EMAIL_REGEX = r'^[a-z0-9][a-z0-9_\.]{4,31}[a-z0-9]@[a-z0-9]{2,}(?:\.[a-z0-9]{2,12}){1,2}$'

EDU_EMAIL_REGEX = r'^[0-9a-z]+@[0-9a-z\.]'

EDGE_AUTO_EMAIL_REGEX = r'@privaterelay.appleid.com|[0-9a-z]+\_autoemail'

# ? EMAIL CONSTRAINTS
_FLATTEN_DOMAIN_LIST = [(x, y['domains']) for (x, y) in EMAIL_DOMAIN_REGEX.items()]
DOMAIN_GROUP_DICT = dict((x, v[0]) for v in _FLATTEN_DOMAIN_LIST for x in v[1])

PRIVATE_EMAIL_DOMAINS = [
    'gmail.com', 'yahoo.com',
    'yahoo.com.vn', 'icloud.com',
    'email.com', 'hotmail.com', 'gmai.com', 'outlook.com']
LEAST_NUM_EMAIL_CHAR = 8
LONGEST_VIETNAMESE_WORD = 'Nghieng'
