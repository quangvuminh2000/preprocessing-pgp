import re


def number_pad_replace(match: re.Match) -> str:
    """
    Replacement function for removing padding in number string

    Parameters
    ----------
    match : re.Match
        match object received by regex

    Returns
    -------
    str
        return string without padding
    """

    number = int(match.group(1))

    return format(number, '01d')
