from typing import Tuple

import pandas as pd
from unidecode import unidecode
from tqdm import tqdm

try:
    try:
        from evaluate import evaluate_true_prediction
    except ImportError:
        from scripts.train_script.evaluate import evaluate_true_prediction
except ImportError:
    from train_script.evaluate import evaluate_true_prediction

# CONFIG
tqdm.pandas()


def find_match_word(base_word: str, name_dict_df: pd.DataFrame) -> str:
    """
    Find the word in the dictionary if not return the word

    Parameters
    ----------
    base_word : str
        The original word to find inside the dictionary
    name_dict_df : pd.DataFrame
        The DF dictionary containing the words and their diacritics ref

    Returns
    -------
    str
        The returned diacritics version or original word if not have
    """
    try:
        return name_dict_df.loc[name_dict_df['without_accent'] == base_word]['with_accent'].values[0]
    except:
        return base_word


def rule_base_middlename(middlename: str,
                         base_middlename: str,
                         name_dict_df: pd.DataFrame) -> str:
    """
    Rule-base to replace middlename when it is changed by the prediction of the model

    Parameters
    ----------
    middlename : str
        The middlename predicted by the model
    base_middlename : str
        The original middlename when input to model
    name_dict_df : pd.DataFrame
        The DF dictionary containing the middlename and their diacritics ref

    Returns
    -------
    str
        The returned rule-based middlename after being preprocessed
    """

    middle_words = middlename.split()
    base_middle_words = base_middlename.split()
    de_middle_words = [unidecode(word) for word in middle_words]

    final_middle_words = base_middle_words
    track_idx = list(range(len(base_middle_words)))
    for de_idx, de_word in enumerate(de_middle_words):
        if de_word in base_middle_words:
            if de_word == base_middle_words[de_idx]:
                final_middle_words[de_idx] = middle_words[de_idx]
            else:
                base_idx = base_middle_words.index(de_word)
                final_middle_words[base_idx] = middle_words[de_idx]

            track_idx.remove(de_idx)

    # All track are visited
    if track_idx == []:
        return ' '.join(final_middle_words)

    # Some word in base is not visited
    for idx in track_idx:
        final_middle_words[idx] = find_match_word(
            base_middle_words[idx], name_dict_df)
    return ' '.join(final_middle_words)


def rule_base_word(word: str, word_base: str, name_dict_df: pd.DataFrame) -> str:
    """
    Apply rule-base to one word

    Parameters
    ----------
    word : str
        The predicted word
    word_base : str
        The original input word
    name_dict_df : pd.DataFrame
        The dictionary DF to track for name

    Returns
    -------
    str
        The best name found in the dictionary
    """
    de_word = unidecode(word)
    if (de_word != word_base):
        return find_match_word(word_base, name_dict_df)
    return word


def rule_base_name(name: str, base_name: str, name_dicts: Tuple) -> str:
    # extract name_dicts
    firstname_dict_df, middlename_dict_df, lastname_dict_df = name_dicts

    # split first, middle, last name
    firstname = name.split()[-1]
    base_firstname = base_name.split()[-1]
    middlename = ' '.join(name.split()[1:-1])
    base_middlename = ' '.join(base_name.split()[1:-1])
    lastname = name.split()[0]
    base_lastname = base_name.split()[0]

    # take firstname when 1 word
    if firstname == lastname and base_firstname == base_lastname:
        return rule_base_word(firstname, base_firstname, firstname_dict_df)

    # applying rule-base
    rule_firstname = rule_base_word(
        firstname, base_firstname, firstname_dict_df)
    rule_middlename = rule_base_middlename(
        middlename, base_middlename, middlename_dict_df)
    rule_lastname = rule_base_word(lastname, base_lastname, lastname_dict_df)

    # joining categories to make full name
    if rule_middlename == '':
        fullname = ' '.join([rule_lastname, rule_firstname])
    else:
        fullname = ' '.join([rule_lastname, rule_middlename, rule_firstname])
    fullname = fullname.replace(r'\s+', ' ').strip()
    return fullname


def rulebase_postprocess(df: pd.DataFrame, name_dicts: Tuple):
    """
    Process rulebase of VN names on predicted DF and return new DF containing the rulebase names

    Parameters
    ----------
    df : pd.DataFrame
        The predicted DF
    name_dicts : Tuple
        The tuple of name dicts : first, middle, last

    Returns
    -------
    pd.DataFrame, float
        The returned DF containing the rulebase names and the predicted score
    """

    rule_base_df = df[['with_accent', 'without_accent', 'predict']].copy()

    rule_base_df['pred_rule_base'] = rule_base_df.progress_apply(
        lambda row: rule_base_name(
            row['predict'], row['without_accent'], name_dicts),
        axis=1
    )

    score = evaluate_true_prediction(
        rule_base_df, accent_col='with_accent', pred_col='pred_rule_base')

    return rule_base_df, score


if __name__ == '__main__':
    rule_base_df, score = rulebase_postprocess(pd.DataFrame({
        'with_accent': ['Văn Sỹ Hoàng', 'Trần Thảo Minh Trang', 'Tú Riêng'],
        'without_accent': ['Van Sy Hoang', 'Tran Thao Minh Trang', 'Tu Rieng'],
        'predict': ['Văn Sỹ Hoàng', 'Trần Thảo Minh Trang', 'Từ Ngủ']
    }))

    print(rule_base_df)
