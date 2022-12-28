import stanza
import pandas as pd
from stanza import DownloadMethod
from unidecode import unidecode
from tqdm import tqdm
from multiprocesspandas import applyparallel

try:
    from const import NON_HUMAN_REG_LIST, REPLACE_HUMAN_REG_DICT
except ImportError:
    from ...const import NON_HUMAN_REG_LIST, REPLACE_HUMAN_REG_DICT

tqdm.pandas()


class HumanNameExtractor:
    def __init__(self, language: str = 'vi'):
        self.language = language
        self.load_nlp()

    def load_nlp(self):
        try:
            self.nlp = stanza.Pipeline(
                self.language, download_method=DownloadMethod.REUSE_RESOURCES, verbose=0, use_gpu=False)
        except:
            stanza.download(self.language, verbose=0)
            self.nlp = stanza.Pipeline(
                self.language, download_method=DownloadMethod.REUSE_RESOURCES, verbose=0, use_gpu=False)

    def check_contains_label(self, sentence: str, label: str = 'PERSON'):
        if not hasattr(self, 'nlp') or self.nlp is None:
            raise TypeError(
                'Please load the nlp model because this function requires it to execute')

        doc = self.nlp(sentence)
        sentence = doc.sentences[0]

        sent_types = [s.type for s in sentence.ents]
        return label in sent_types

    def extract_first_person(self, name: str):
        """
        Extract and return the first person name found in the sentence provided

        Parameters
        ----------
        name : str
            The provided name which can contains non-human names

        Returns
        -------
        str
            The first name appears in the sentence or empty string if not found any

        Raises
        ------
        TypeError
            If not having nlp loaded -> Please use load_nlp() method to load the nlp
        """
        names = self.extract_persons(name)

        return names[0] if names != [] else ''

    def extract_persons(self, name: str):
        """
        Extract and return the first person name found in the sentence provided

        Parameters
        ----------
        name : str
            The provided name which can contains non-human names

        Returns
        -------
        List[str]
            All the person's names appear in the sentence or empty list if not found any

        Raises
        ------
        TypeError
            If not having nlp loaded -> Please use load_nlp() method to load the nlp
        """
        if not hasattr(self, 'nlp') or self.nlp is None:
            raise TypeError(
                'Please load the nlp model because this function requires it to execute')

        doc = self.nlp(name)
        sentence = doc.sentences[0]

        # Extract the person name from the predicted text
        names = [s.text for s in sentence.ents if s.type == 'PERSON']

        return names


def replace_non_human_reg(name: str) -> str:
    for word, to_word in REPLACE_HUMAN_REG_DICT.items():
        name = name.replace(word, to_word)
    return name.strip()


def remove_non_person_with_rule(
    df: pd.DataFrame,
    name_col: str
) -> pd.DataFrame:
    print('Creating new non-accent names...')
    clean_df = df.copy()
    clean_df['without_accent'] = clean_df[name_col].progress_apply(unidecode)

    # Process with rule to remove non_human_names
    print('Cleaning special cases...')
    non_clean_mask = clean_df['without_accent'].progress_apply(
        lambda name: any(substr in name for substr in NON_HUMAN_REG_LIST))
    clean_df = clean_df[~non_clean_mask]
    clean_df = clean_df.reset_index(drop=True)
    print('\n\n')
    print('-'*20)
    print(f'{non_clean_mask.sum()} non human names detected with rule!')
    print('-'*20)
    print('\n\n')

    # Process to replace other non_meaning parts
    print('Replacing non-meaningful names...')
    clean_df[f'replaced_{name_col}'] = clean_df[name_col].progress_apply(
        replace_non_human_reg)

    replaced_mask = clean_df[f'replaced_{name_col}'] != clean_df[name_col]

    print('\n\n')
    print('-'*20)
    print(f'{replaced_mask.sum()} names replaced to beautify with rule!')
    print('-'*20)
    print('\n\n')

    # Clean up after working
    clean_df = clean_df.drop(columns=['without_accent', name_col])
    clean_df = clean_df.rename(columns={f'replaced_{name_col}': name_col})

    return clean_df, df[non_clean_mask]


def remove_non_person(df: pd.DataFrame,
                      name_col: str,
                      model: HumanNameExtractor,
                      multiprocessing: bool = False,
                      n_cpu: int = None,
                      ):
    """
    Remove the non-person names from the list of names

    Parameters
    ----------
    df : pd.DataFrame
        The DF containing the names
    name_col : str
        The name of the column containing the names
    model : HumanNameExtractor
        The model object of HumanNameExtractor class
    multiprocessing : bool
        Whether to use the multiprocessing cores, by default False
    n_cpu : int
        The number of cpu is required when enabled multiprocessing, by default None

    Returns
    -------
    Tuple
        The returned tuple of DFs containing human names and non-human names
    """
    print('Removing non-person names...')
    if multiprocessing:
        assert n_cpu != None, "Please provide number of cpu!"
        names = df[name_col].apply_parallel(
            model.extract_first_person, num_processes=n_cpu)
    else:
        names = df[name_col].progress_apply(model.extract_first_person)

    null_name_mask = names == ''

    clean_df = df[~null_name_mask].copy()
    clean_df[name_col] = names[~null_name_mask]

    clean_df = clean_df.reset_index(drop=True)

    # Remove non-person with rule
    print('\n\nRemoving non-person names with rule...')
    clean_df, non_clean_df = remove_non_person_with_rule(clean_df, name_col)

    non_clean_df = pd.concat(
        [df[null_name_mask], non_clean_df], ignore_index=False).sort_index()

    return clean_df, non_clean_df


if __name__ == '__main__':
    human_extract = HumanNameExtractor(language='vi')
    input_sent = input("Please input something: ")
    print(human_extract.extract_first_person(input_sent))
