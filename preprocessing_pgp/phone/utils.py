import re


def basic_phone_preprocess(phone: str) -> str:
    """
    Perform basic preprocessing on phone

    1. Remove all non-digit from phone string
    2. Remove all the spaces from phone string

    Parameters
    ----------
    phone : str
        The input phone to preprocess

    Returns
    -------
    str
        Preprocessed phone
    """

    clean_phone = re.sub(r"[^0-9]", "", phone)
    clean_phone = re.sub(r"\s+", "", phone)

    return clean_phone
