import argparse
import os
from dotenv import load_dotenv, find_dotenv
import util

root_path = os.path.abspath("")
questionnaire_path = os.path.join(root_path, "..", "..", "data", "raw",
                                  "questionnaire.csv")


def get_instagram_credentials():
    env_variables_path = os.path.join(root_path, "..", "..",
                                      "instagram_access.env")
    dotenv_path = find_dotenv(env_variables_path)
    load_dotenv(dotenv_path)
    instagram_username = os.environ.get("INSTAGRAM_USERNAME")
    instagram_password = os.environ.get("INSTAGRAM_PASSWORD")

    return instagram_username, instagram_password


parser = argparse.ArgumentParser(
    description=""" This is a simple algorithm to scrape Instagram
                    User data. This algorithm downloads data from
                    ReadOrSee/data/raw usersnames, excluding
                    duplicates.""")

parser.add_argument("--last", help="""This will continue the data
                                    scraping from the last downloaded
                                    user.""",
                    action="store_true",
                    required=True)

args = parser.parse_args()

name, pw = get_instagram_credentials()

print(util.get_downloaded_users_list())
