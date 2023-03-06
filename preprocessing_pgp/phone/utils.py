import re


def basic_phone_preprocess(phone: str) -> str:
    """
    Perform basic preprocessing on phone

    1. Remove all non-digit from phone string
    2. Remove all the spaces from phone string
    3. Remove (84) from head phone
    4. Add 0 to number not starting with 0

    Parameters
    ----------
    phone : str
        The input phone to preprocess

    Returns
    -------
    str
        Preprocessed phone
    """
    if len(phone) < 9:
        return phone

    clean_phone = re.sub(r"[^0-9]", "", phone)
    clean_phone = re.sub(r"\s+", "", phone)
    if len(phone) != 9:
        clean_phone = re.sub(r"(?i)^84", "0", phone)
    if clean_phone[0] != '0':
        clean_phone = '0' + clean_phone

    return clean_phone
