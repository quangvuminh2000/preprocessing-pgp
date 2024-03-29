from preprocessing_pgp.phone.const import (
    SUB_PHONE_11NUM,
    SUB_TELEPHONE_10NUM,
    DICT_4_SUB_PHONE,
    DICT_4_SUB_TELEPHONE,
    DICT_NEW_MOBI_PHONE_VENDOR,
    DICT_NEW_TELEPHONE_VENDOR
)


def convert_mobi_phone(phone: str) -> str:
    """
    Convert old mobiphone format(11 numbers) into new format(10 numbers)

    Parameters
    ----------
    phone : str
        The old format phone

    Returns
    -------
    str
        Newly returned phone with new format
    """
    if phone[:4] in SUB_PHONE_11NUM:
        return DICT_4_SUB_PHONE[phone[:4]] + phone[4:]
    else:
        return None


def convert_phone_region(old_region: str) -> str:
    """
    Convert old region code phone to new region code phone

    Parameters
    ----------
    old_region : str
        Phone from old region

    Returns
    -------
    str
        New phone from new region
    """
    if old_region[:2] in SUB_TELEPHONE_10NUM:
        return DICT_4_SUB_TELEPHONE[old_region[:2]] + old_region[2:]

    if old_region[:3] in SUB_TELEPHONE_10NUM:
        return DICT_4_SUB_TELEPHONE[old_region[:3]] + old_region[3:]

    if old_region[:4] in SUB_TELEPHONE_10NUM:
        return DICT_4_SUB_TELEPHONE[old_region[:4]] + old_region[4:]

    return None


def convert_mobi_phone_vendor(phone: str) -> str:
    """
    Convert valid mobi phone to its vendor

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

def convert_tele_phone_vendor(phone: str) -> str:
    """
    Convert valid tele phone to its vendor

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
