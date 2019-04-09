import os


def get_downloaded_users_list():
    root_path = os.path.abspath("")
    data_path = os.path.join(root_path, "..", "..", "data", "external")
    return os.listdir(data_path)
