
# import matplotlib.pyplot as plt
# import seaborn as sns
import pandas as pd
# import wordcloud as wc
from unidecode import unidecode
from tqdm import tqdm

tqdm.pandas()


def syntax_plot(df_old,
                df_new,
                name_col='Name',
                correct_col='Corrected',
                wrong_col='Wrong',
                savefile=False,
                filepath=None):
    """
    Plotting the bar chart of the differences in the old DF and new DF

    Parameters
    ----------
    df_old : pd.DataFrame
        The old data frame with a column of 'Name' representing the names of the person
    df_new : pd.DataFrame
        The new data frame with a column of 'Name' representing the names of the person
    name_col : str, optional
        The name of the column which contains the names, by default 'Name'
    corrected_col : str, optional
        The name of the column which remains the same, by default 'Corrected'
    wrong_col : str, optional
        The name of the column which changes, by default 'Wrong'
    savefile : bool, optional
        Whether to save figure file, by default False
    filepath : str, optional
        If so provide name of the file or path inside folder plots, by default None
    """

    mask = df_old[name_col] == df_new[name_col]
    n_corrected = df_old[mask].shape[0]
    n_wrong = df_old.shape[0] - n_corrected

    n_df = pd.DataFrame({
        'Type': [correct_col, wrong_col],
        'Number': [n_corrected, n_wrong]
    })

    # Plot the barplot to show the differences
    fig, ax = plt.subplots(figsize=(10, 12))
    ax = sns.barplot(x='Type', y='Number', data=n_df)
    ax.bar_label(ax.containers[0])
    sns.despine(fig)

    if savefile:
        assert filepath != None, 'Please provide the filename when savefile=True'
        plt.savefig(filepath)


def pandas_diff_barplot(df: pd.DataFrame,
                        first_column: str,
                        second_column: str,
                        correct_col: str = 'Corrected',
                        wrong_col: str = 'Wrong',
                        savefile: bool = False,
                        filepath: str = None):
    """
    Plotting differences in a DF containing 2 columns

    Parameters
    ----------
    df : pd.DataFrame
        The DF containing at least 2 columns
    first_column : str
        First column to seek for differences
    second_column : str
        Second column to seek for differences
    correct_col : str, optional
        The output column name for the same, by default 'Corrected'
    wrong_col : str, optional
        The output column name for the different, by default 'Wrong'
    savefile : bool, optional
        Whether to save file or not, by default False
    filepath : str, optional
        If savefile=True -> provide filepath, by default None
    """
    mask = df[first_column] == df[second_column]
    n_corrected = df[mask].shape[0]
    n_wrong = df.shape[0] - n_corrected

    n_df = pd.DataFrame({
        'Type': [correct_col, wrong_col],
        'Number': [n_corrected, n_wrong]
    })

    # Plot the barplot to show the differences
    fig, ax = plt.subplots(figsize=(10, 12))
    ax = sns.barplot(x='Type', y='Number', data=n_df)
    ax.bar_label(ax.containers[0])
    sns.despine(fig)

    if savefile:
        assert filepath != None, 'Please provide the filename when savefile=True'
        plt.savefig(filepath)


def wordcloud_plot(df: pd.DataFrame,
                   column: str = 'with_accent',
                   savefile: bool = False,
                   filepath: str = None):

    wordcloud = wc.WordCloud(max_font_size=50, max_words=100,
                             background_color='white').generate(' '.join(df[column]))

    plt.figure()
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')

    if savefile:
        assert filepath != None, 'Please provide the filename when savefile=True'
        wordcloud.to_file(filepath)


def remove_non_accent_names(names_df: pd.DataFrame, name_col='name', remove_single_name=True) -> pd.DataFrame:
    """
    Remove non accent names inside the DF

    Parameters
    ----------
    names_df : pd.DataFrame
        The original names DF
    name_col : str, optional
        The column containing the data of names, by default 'name'
    remove_single_name : bool, optional
        Whether to remove a single word name, by default True

    Returns
    -------
    pd.DataFrame
        The clean final DF without any non_accent name
    """
    print("Decoding names...")
    names = names_df[name_col].copy()
    de_names = names.progress_apply(unidecode)

    with_accent_mask = names != de_names

    clean_names = names[with_accent_mask]
    clean_de_names = de_names[with_accent_mask]

    if not remove_single_name:
        len_name = names.apply(lambda name: len(name.split()))
        one_word_mask = len_name == 1
        clean_names = names[with_accent_mask | one_word_mask]
        clean_de_names = de_names[with_accent_mask | one_word_mask]

    clean_names_df = pd.DataFrame({
        'without_accent': clean_de_names,
        'with_accent': clean_names
    })
    
    without_accent_names_df = names_df[~with_accent_mask].copy()

    return clean_names_df, without_accent_names_df
