"""
Module to extract and replace non-human names
"""
from preprocessing_pgp.name.const import REPLACE_HUMAN_REG_DICT


def replace_non_human_reg(name: str) -> str:
    """
    Replace non-human term with correct term using the dictionary based

    Parameters
    ----------
    name : str
        The name to preprocess replace non-human term

    Returns
    -------
    str
        The clean name containing the transformed non-human names
    """
    for word, to_word in REPLACE_HUMAN_REG_DICT.items():
        name = name.replace(word, to_word)
    return name.strip()
