"""
Module contains constants for email processing
"""

# ? NONE TYPE EMAIL DICT
NAN_EMAIL_LIST = ['nan', 'none', '']

# ? EMAIL REGEX

AT_LEAST_ONE_CHAR_REGEX = r'(?=.*[a-z])'

EMAIL_DOMAIN_REGEX = {
    'gmail': {
        'domains': r'gmail.com([.][a-z]{2})?$',
        'regex': r'^(?!.*google)[a-z][a-z0-9\.]{4,28}[a-z0-9]'
    },
    'yahoo': {
        'domains': r'(yahoo|ymail|myyahoo).com([.][a-z]{2})?$',
        'regex': r'^[a-z][a-z0-9_\.]{2,30}[a-z0-9]'
    },
    'ms': {
        'domains': r'(hotmail|outlook).com([.][a-z]{2})?$',
        'regex': r'^[a-z][a-z0-9-_\.]{0,30}[a-z0-9-_]?'
    },
    'fpt': {
        'domains': r'.*fpt.*$',
        'regex': r'^[a-z0-9][a-z0-9_\.]{2,31}[a-z0-9]'
    }
}

COMMON_EMAIL_REGEX = r'^[a-z0-9][a-z0-9_\.]{2,30}[a-z0-9]@[a-z0-9]{2,}(?:\.[a-z0-9]{2,12}){1,2}$'

EDU_EMAIL_REGEX = r'^[a-z0-9][a-z0-9_.]{4,30}[a-z0-9]'

EDGE_AUTO_EMAIL_REGEX = r'@privaterelay.appleid.com|[0-9a-z]+\_autoemail'

# ? EMAIL CONSTRAINTS
DOMAIN_GROUP_DICT = dict((reg_li['domains'], domain) for domain, reg_li in EMAIL_DOMAIN_REGEX.items())

PRIVATE_EMAIL_DOMAINS = [
    'gmail.com', 'yahoo.com',
    'yahoo.com.vn', 'icloud.com',
    'email.com', 'hotmail.com', 'gmai.com', 'outlook.com']
LEAST_NUM_EMAIL_CHAR = 8
LONGEST_VIETNAMESE_WORD = 'Nghieng'
