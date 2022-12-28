
import pandas as pd
from tqdm import tqdm


def n_gram_name(name: str, n_gram: int = 1, remove_lastname: bool = False):
    """
    Create n-grams for a certain name

    Parameters
    ----------
    name : str
        The full name of a person
    n_gram : int, optional
        The number of word each gram, by default 1
    remove_lastname : bool, optional
        Whether to take the lastname, by default False

    Returns
    -------
    list[str]
        n-gram list of string of the name
    """

    words = name.split(' ')

    # Remove last name
    if remove_lastname:
        words = words[1:]

    gram_names = []
    for i in range(len(words)-n_gram+1):
        gram_names.append(' '.join(words[i:i+n_gram]))

    return gram_names


def process_n_gram(name_series: pd.Series, n_gram: int = 1):
    """
    Processing n-gram on a single series

    Parameters
    ----------
    name_series : pd.Series
        Input name series to process
    n_gram : int, optional
        The number of word each gram, by default 1

    Returns
    -------
    list
        Return list of names
    """

    names_1_gram = []
    for name in tqdm(name_series):
        names_1_gram += [name] + n_gram_name(name, n_gram=n_gram)

    return names_1_gram


def aug_n_gram(data_path: str,
               n_gram: int = 1,
               save_file: bool = False,
               file_save_path: str = None) -> pd.DataFrame:
    """
    Process & save DataFrame with n-gram words

    Parameters
    ----------
    data_path : str
        The path to dataset to preprocess (.parquet)
    n_gram : int, optional
        The number of word each gram, by default 1
    save_file : bool, optional
        Whether to save data file or not, by default False
    file_save_path : str, optional
        If save_file=True then define the path to save the file, by default None

    Returns
    -------
    pd.DataFrame
        n-gram DataFrame after processing
    """

    name_df = pd.read_parquet(data_path)
    print(f'Doing n-gram on data of shape: {name_df.shape}\n')

    with_accent_names = name_df['with_accent']
    without_accent_names = name_df['without_accent']

    # Do n-gram and save to new DF
    with_accent_names_1_gram = process_n_gram(with_accent_names, n_gram=n_gram)
    without_accent_names_1_gram = process_n_gram(
        without_accent_names, n_gram=n_gram)

    # Create n-gram DF
    name_1_gram_df = pd.DataFrame({
        'with_accent_1_gram': with_accent_names_1_gram,
        'without_accent_1_gram': without_accent_names_1_gram
    })
    print(f'Data shape after n-gram: {name_1_gram_df.shape}\n')
    print('Data after n-gram:')
    print(name_1_gram_df.head(5))

    # Save datafile
    if save_file:
        print(f'Saving data to {file_save_path}')
        assert file_save_path != None, 'Please provide the filename when save_file=True'
        name_1_gram_df.to_parquet(file_save_path)

    return name_1_gram_df


if __name__ == '__main__':
    my_name = 'Vũ Minh Quang'
    print(my_name)
    print(n_gram_name(my_name, n_gram=1, remove_lastname=True))

    # Test on the dataframe
    name_df = pd.DataFrame({
        'Name': ['Vũ Minh Quang']
    })
