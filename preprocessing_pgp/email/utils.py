"""
Module contains utility functions to help the email's functions
"""

from typing import List
from unidecode import unidecode


def split_email(email: str) -> List[str]:
    """
    Split email into email's name & group (default by `@`)

    Parameters
    ----------
    email : str
        The original email

    Returns
    -------
    List[str]
        The list contains email's `name` and `group`
    """
    if not email:
        return [None, None]

    split_result = email.split('@', maxsplit=1)

    if len(split_result) == 2:
        return split_result

    return [*split_result, None]

def is_name_accented(name: str) -> bool:
    """
    Check whether the name is accented or not

    Parameters
    ----------
    name : str
        The input name

    Returns
    -------
    bool
        Whether the name is accented
    """
    return unidecode(name) != name

def clean_email_name(name: str) -> str:
    """
    Process cleaning email name -- removing '.' and other processing

    Parameters
    ----------
    name : str
        The input email's name

    Returns
    -------
    str
        Cleaned email name
    """
    cleaned_name = name.replace('.', '')
    cleaned_name = cleaned_name.lower()

    return cleaned_name
