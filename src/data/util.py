import os
from dotenv import load_dotenv, find_dotenv
import pandas as pd
import re
import numpy as np


def scraped_users_list():
    """ Return a set of scrapped Instagram usernames. """
    root_path = os.path.abspath("")
    data_path = os.path.join(root_path, "..", "..", "data", "external")
    return set(os.listdir(data_path))


def load_env_variables():
    """ Load environment variables in file to module. """
    root_path = os.path.abspath("")
    env_variables_path = os.path.join(root_path, "..", "..",
                                      "instagram_access.env")
    dotenv_path = find_dotenv(env_variables_path)
    load_dotenv(dotenv_path)


def get_instagram_credentials():
    """ Return Instagram credentials (username, password). """
    instagram_username = os.environ.get("INSTAGRAM_USERNAME")
    instagram_password = os.environ.get("INSTAGRAM_PASSWORD")
    return instagram_username, instagram_password


def non_scraped_instagram_users():
    """ Returns a list of instagram usrenames not scraped yet. """
    scraped_usernames = scraped_users_list()
    all_usernames = _get_all_instagram_users()
    return all_usernames - scraped_usernames


def _get_all_instagram_users():
    """ Returns a set of all instagram usernames. """
    instagram_df, _ = preprocess_questionnaire(load_questionnaire())
    return set(instagram_df["instagram_user_name"])

# ######## Questionnaire Normalization Process #########


def load_questionnaire():
    """ Return the questionnaire DataFrame. """
    root_path = os.path.abspath("")
    data_path = os.path.join(root_path, "..", "..",
                             "data", "raw", "questionnaire.csv")
    df = pd.read_csv(data_path, encoding="utf-8")
    return df


def _get_number(choice):
    pattern = re.compile(r"(\d)(\.|\w)", re.UNICODE)
    match = re.match(pattern, choice)
    return int(match.group(1))


def _salary_normalization(text):
    if text == "Até R$ 1.874,00":
        return "E"
    elif text == "R$ 1.874,01 a R$ 3.748,00":
        return "D"
    elif text == "R$ 3.748,01 a R$ 9.370,00":
        return "C"
    elif text == "R$ 9.370,01 a R$ 18.740,00":
        return "B"
    else:
        return "A"


def _rename_columns(columns):
    """ Return a list of new column names. """
    new_columns = columns.copy()
    new_columns[0] = "form_application_date"
    new_columns[1] = "email"
    new_columns[5] = "sex"
    new_columns[6] = "birth_date"
    new_columns[7] = "household_income"
    new_columns[9] = "facebook_hours"
    new_columns[10] = "twitter_hours"
    new_columns[11] = "instagram_hours"
    new_columns[12] = "academic_degree"
    new_columns[13] = "course_name"
    new_columns[14] = "semesters"
    new_columns[15] = "scholarship"
    new_columns[16] = "accommodation"
    new_columns[17] = "works"
    new_columns[20] = "depression_diagnosed"
    new_columns[21] = "in_therapy"
    new_columns[22] = "antidepressants"
    new_columns[45] = "twitter_user_name"
    new_columns[46] = "instagram_user_name"

    return new_columns


def _normalize_usernames(df):
    df["instagram_user_name"] = (df["instagram_user_name"]
                                 .map(lambda x: str(x).lower()
                                 .replace("@", "").strip()))
    df["twitter_user_name"] = (df["twitter_user_name"]
                               .map(lambda x: str(x).lower()
                               .replace("@", "").strip()))
    return df


def _remove_duplicates(df):
    pass


def _remove_malformed_usernames(df, column):
    """ Return a new DataFrame without
    Instagram and Twitter malformed unames.

    Keyword arguments:
    df -- DataFrame
    column -- Username column to check
    """
    def check_validity(uname):
        """ Return True if uname is a valid username. False otherwise. """
        uname = uname.strip()
        if " " in uname:
            return False
        elif not uname:
            return False
        elif uname.isdigit():
            return False
        return True

    df = df[pd.notnull(df[column])]
    mask = list(map(lambda x: check_validity(x), df[column]))
    df = df[mask]
    df = _normalize_usernames(df)
    df = df.drop_duplicates(subset=column)
    return df


def _save_dataframe(dataframe, filename):
    """ Save the dataframe with the filename name. """
    root_path = os.path.abspath("")
    data_path = os.path.join(root_path, "..", "..",
                             "data", "interim", filename)
    dataframe.to_csv(data_path, encoding="utf-8", index=False)


def preprocess_questionnaire(df):
    """ Pre-process the questionnaire data and return it.

    Side effects:
    1. The Instagram dataframe in 'interim' folder is updated
    2. The Twitter dataframe in 'interim' folder is updated
    3. The General dataframe in 'interim' folder without any row
        specially excluded by twitter or instagram username malformation

    We have Instagram and Twitter dataframes because of username malformation.
    Since it is possible for users to input malformed usernames while answering
    the questionnaire, we remove answers (rows) with malformed usernames.

    Return:
    instagram_df -- Instagram dataframe
    twitter_df -- Twitter dataframe
    """
    df.iloc[:, 24:45] = df.iloc[:, 24:45].applymap(_get_number)
    df["BDI"] = df.iloc[:, 24:45].apply(np.sum, axis=1)
    df.iloc[:, 7] = df.iloc[:, 7].transform(_salary_normalization)

    new_columns = _rename_columns(df.columns.values)
    df.columns = new_columns

    bdi_range = list(range(24, 45))
    drop_columns = [2, 3, 4, 8, 18, 19, 23, 47] + bdi_range

    df = df.drop(df.columns[drop_columns], axis=1)

    df["form_application_date"] = (pd.to_datetime(df["form_application_date"]))
    df["form_application_date"] = df["form_application_date"].apply(
        lambda x: x.date)
    df["email"] = df["email"].transform(lambda x: x.lower())

    _save_dataframe(df, "all_participants.csv")

    instagram_df = df.copy()
    twitter_df = df.copy()

    instagram_df = _remove_malformed_usernames(instagram_df,
                                               "instagram_user_name")
    twitter_df = _remove_malformed_usernames(twitter_df, "twitter_user_name")

    _save_dataframe(instagram_df, "instagram.csv")
    _save_dataframe(twitter_df, "twitter.csv")

    return instagram_df, twitter_df
