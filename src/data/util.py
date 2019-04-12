import os
from dotenv import load_dotenv, find_dotenv
from preprocessing import RawPreProcess


def scraped_users_list():
    """ Return a set of scrapped Instagram usernames. """
    root_path = os.path.abspath("")
    data_path = os.path.join(root_path, "..", "..", "data", "external",
                             "instagram")
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
    """ Returns a set of all valid instagram usernames.

    Side effect:
    1. Update the interim/ folder with new preprocessed data
    """
    rpp = RawPreProcess()
    instagram_df, _ = rpp.preprocess()
    rpp.save_processed()
    return set(instagram_df["instagram_user_name"])
