"""
Module to detect for special phones and meanings in phones
"""
from preprocessing_pgp.phone.const import (
    DICT_NEW_MOBI_PHONE_VENDOR,
    DICT_NEW_TELEPHONE_VENDOR,
    DICT_PHONE_TAIL_TYPE
)


def detect_mobi_phone_vendor(phone: str) -> str:
    """
    Detect valid mobi phone's vendor

    Parameters
    ----------
    phone : str
        Valid mobi phone

    Returns
    -------
    str
        Mobi phone vendor
    """

    if phone[:3] in DICT_NEW_MOBI_PHONE_VENDOR.keys():
        return DICT_NEW_MOBI_PHONE_VENDOR[phone[:3]]

    return None


def detect_tele_phone_vendor(phone: str) -> str:
    """
    Detect valid tele phone's vendor

    Parameters
    ----------
    phone : str
        Valid tele phone

    Returns
    -------
    str
        Tele phone vendor
    """
    if phone[:4] in DICT_NEW_TELEPHONE_VENDOR.keys():
        return DICT_NEW_TELEPHONE_VENDOR[phone[:4]]

    if phone[:3] in DICT_NEW_TELEPHONE_VENDOR.keys():
        return DICT_NEW_TELEPHONE_VENDOR[phone[:3]]

    return None


def get_phone_tail_type(
    phone: str,
    tail_size: int
) -> str:
    """
    Get the tail type of phone with the given size

    Parameters
    ----------
    phone : str
        The valid phone number
    tail_size : int
        The size of the tail

    Returns
    -------
    str
        The type of the phone tail
    """
    return DICT_PHONE_TAIL_TYPE.get(phone[-tail_size:], None)


def detect_meaningful_phone(phone: str) -> str:
    """
    Detect the type of meaningful phone

    Parameters
    ----------
    phone : str
        Valid phone number

    Returns
    -------
    str
        Type of meaningful phone
    """

    # ? 6-3 last numbers
    for tail_size in range(6, 2, -1):
        tail_num_type = get_phone_tail_type(phone, tail_size)
        if tail_num_type is not None:
            return tail_num_type

    return "Số Thường"
