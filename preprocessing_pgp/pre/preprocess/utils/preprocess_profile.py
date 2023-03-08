"""
Module contains function to preprocess profile
"""
import pandas as pd


from preprocessing_pgp.name.preprocess import preprocess_df
from preprocessing_pgp.utils import parallelize_dataframe
from preprocessing_pgp.name.split_name import NameProcess


def remove_same_username_email(
    data: pd.DataFrame,
    name_col: str = 'name',
    email_col: str = 'email',
) -> pd.DataFrame:
    """
    Filter out the rows with email's username is customer's name
    """
    data['username_email'] = data[email_col].str.split('@').str[0]
    data.loc[
        data[name_col] == data['username_email'],
        name_col
    ] = None

    data = data.drop(columns=['username_email'])

    return data


def extracting_pronoun_from_name(
    data: pd.DataFrame,
    condition: pd.Series,
    name_col: str = 'name',
) -> pd.DataFrame:
    """
    Cleansing and Extracting pronoun from name
    """
    name_process = NameProcess()
    data.loc[
        condition,
        ['clean_name', 'pronoun']
    ] = data.loc[
        condition,
        name_col
    ].apply(name_process.CleanName).tolist()

    data.loc[
        condition,
        name_col
    ] = data.loc[
        condition,
        'clean_name'
    ]
    data = data.drop(columns=['clean_name'])

    return data


# ? MAIN FUNCTIONS FOR PREPROCESS PROFILE
def cleansing_profile_name(
    data: pd.DataFrame,
    name_col: str = 'name',
    n_cores: int = 1
) -> pd.DataFrame:
    """
    Apply cleansing to name in profile
    """

    cleansed_data = parallelize_dataframe(
        data,
        preprocess_df,
        n_cores=n_cores,
        name_col=name_col
    )

    return cleansed_data


def extra_cleansing_name(
    data: pd.DataFrame,
    customer_condition: pd.Series,
    remove_username_email: bool = True,
    name_col: str = 'name',
    email_col: str = 'email',
    n_cores: int = 1
) -> pd.DataFrame:
    """
    Extra cleansing names
    """
    if remove_username_email:
        print("Removing Same Customer Name with Email Name")
        data = parallelize_dataframe(
            data,
            remove_same_username_email,
            n_cores=n_cores,
            email_col=email_col,
            name_col=name_col
        )

    condition = (data[name_col].notna()) & (customer_condition)

    print(">>> Extracting Pronoun from Name")
    data = parallelize_dataframe(
        data,
        extracting_pronoun_from_name,
        n_cores=n_cores,
        condition=condition,
        name_col=name_col
    )

    return data
