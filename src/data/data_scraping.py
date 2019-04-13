import argparse
import controller
from subprocess import Popen
import os


def scrape_data(login, password, u_file, destination_path):
    """ Create a console that runs the cmd_string command to scrape data. """
    cmd_string = 'instagram-scraper -f {0} -d {1} -n -u {2} -p {3} \
    --retry-forever --media-metadata -t image --profile-metadata'. \
                 format(u_file, destination_path, login, password)

    cmd_string = cmd_string + r' -T {urlname}'

    process = Popen(cmd_string, shell=True)
    process.wait()


def create_temp_file(usernames):
    """ Create temporary file with Instagram usernames.

    We need this in order to feed the cmd_string for as the line argument,
    since it needs a file path.

    Return:
    path -- path of the created file
    """
    with open(".temp", "w") as f:
        for item in usernames:
            f.write("%s\n" % item)
    return ".temp"


def delete_temp_file(file):
    os.remove(file)


def main():
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

    if args.last:
        controller.load_env_variables()
        login, password = controller.get_instagram_credentials()
        usernames_to_scrape = controller.non_scraped_instagram_users()
        u_file = create_temp_file(usernames_to_scrape)
        destination_path = os.path.join(os.path.abspath(""), "..", "..", 
                                        "data", "external", "instagram")
        try:
            scrape_data(login, password, u_file, destination_path)
            delete_temp_file(u_file)
        except KeyboardInterrupt:
            delete_temp_file(u_file)
    else:
        print("Nothing to update.")


if __name__ == "__main__":
    main()
