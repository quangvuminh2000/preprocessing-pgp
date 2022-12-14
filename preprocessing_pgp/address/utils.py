import re
from typing import List


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


def is_empty_string(string: str) -> bool:
    """
    Check if the string is empty or not
    """
    return string == ''


def create_dependent_query(*dependents) -> str:
    """
    Make a `pd.DataFrame` query to match all dependents

    * `dependents` should be strings of comparison query
    """
    query = ''
    if len(dependents) > 0:
        for dep in dependents:
            if len(dep) > 0:
                query = f'{query} & {dep}'\
                    if query != ''\
                    else dep

    return query


def flatten_list(lst: List) -> List:
    """
    Helper function to flatten the list of list into simpler list
    """
    flat_list = [item for sublist in lst for item in sublist]

    return flat_list


def remove_substr(string: str, substr: str) -> str:
    """
    Remove sub-string from original `string` and return the modified string

    Parameters
    ----------
    string : str
        The string to remove sub-string
    substr : str
        The sub-string in string to be removed

    Returns
    -------
    str
        The modified string without sub-string,
        if not having any sub-str found or substring is empty
        return the origin string
    """
    if None in [string, substr]:
        return None

    start_sub_idx = string.rfind(substr)

    if start_sub_idx == -1 or len(substr) == 0:  # * Failure if s
        return string

    end_sub_idx = start_sub_idx + len(substr)

    ret_str = string[:start_sub_idx] + string[end_sub_idx:]

    return ret_str.strip()
