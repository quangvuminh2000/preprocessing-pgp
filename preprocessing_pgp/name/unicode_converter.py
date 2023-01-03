"""
Module to normalized input string to unicode dung san (NFC)
"""
import unicodedata


def minimal_convert_unicode(txt: str) -> str:
    """
    Convert to NFC - Unicode Composed

    Parameters
    ----------
    txt : str
        The input sentence

    Returns
    -------
    str
        The output sentence converted to NFC format
    """
    normalized_txt = unicodedata.normalize('NFKC', txt)

    return normalized_txt.encode().decode()
