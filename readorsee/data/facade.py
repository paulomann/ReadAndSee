import os
from dotenv import load_dotenv, find_dotenv
import pickle
from readorsee.data import preprocessing
from readorsee.data import config
from readorsee.data import stratification


class PreProcessFacade():

    def __init__(self, process_method, save=True):
        if process_method == "complete":
            self._pipeline = [preprocessing.RawPreProcess(),
                              preprocessing.InstagramExternalPreProcess()]
        elif process_method == "raw":
            self._pipeline = [preprocessing.RawPreProcess()]
        elif process_method == "instagram_external":
            self._pipeline = [preprocessing.InstagramExternalPreProcess()]
        elif process_method == "twitter_external":
            self._pipeline = [preprocessing.TweetsExternalPreProcess()]
        else:
            raise ValueError
        self._save = save

    def process_pipeline(self):
        """ Preprocess pipeline and return the last generated dataset in
        the pipeline

        Return:
            data -- if the process is successful, the data will be returned.
                    if save = False, then no data will be returned.
        """
        final_dataset = None
        for pre_process in self._pipeline:
            pre_process.preprocess()
            if self._save:
                final_dataset = pre_process.save_processed()
        return final_dataset


class StratifyFacade():

    def __init__(self, algorithm):
        self._data = {}
        self._algorithm = algorithm

    def stratify(self, tr_size=0.6, val_size=0.2, te_size=0.2,
                 n_sets=10, days=[60, 212, 365], save=True):
        """ Stratify data according to params """
        self._data = {}

        pprocess_facade = PreProcessFacade("complete", save)
        data = pprocess_facade.process_pipeline()
        participants = data["participants"]

        if self._algorithm == "local_search":
            stratification_algorithm = stratification.LocalSearch(
                tr_size, val_size, te_size)

        for d in days:
            attr_name = "data_" + str(d)
            self._data[attr_name] = []
            for n in range(n_sets):
                print("Dataset {} : ".format(n), end="")
                stratified = stratification_algorithm.stratify(
                    participants, d)
                self._data[attr_name].append(stratified)

        return self._data

    def save(self):
        stratified_path = os.path.join(config.PATH_TO_PROCESSED_DATA,
                                       "stratified_data.pickle")
        if self._data.keys():
            with open(stratified_path, "wb") as f:
                pickle.dump(self._data, f)
        else:
            raise ValueError

        self._data

    @staticmethod
    def load_stratified_data():
        stratified_path = os.path.join(config.PATH_TO_PROCESSED_DATA,
                                       "stratified_data.pickle")

        data = None
        with open(stratified_path, "rb") as f:
            data = pickle.load(f)
        return data


class InstagramScraperFacade:

    def __init__(self):
        self._load_env_variables()
        self._login, self._password = self._get_instagram_credentials()

    @property
    def login(self):
        return self._login

    @property
    def password(self):
        return self._password

    def _load_env_variables(self):
        """ Load environment variables in file to module. """
        env_variables_path = config.ENV_VARIABLES
        dotenv_path = find_dotenv(env_variables_path)
        load_dotenv(dotenv_path)

    def _get_instagram_credentials(self):
        """ Return Instagram credentials (username, password). """
        instagram_username = os.environ.get("INSTAGRAM_USERNAME")
        instagram_password = os.environ.get("INSTAGRAM_PASSWORD")
        return instagram_username, instagram_password

    def get_non_scraped_users(self):
        """ Returns a list of instagram usrenames not scraped yet. """
        scraped_usernames = self._scraped_users_list()
        all_usernames = self._get_all_instagram_users()
        return all_usernames - scraped_usernames

    def _scraped_users_list(self):
        """ Return a set of scrapped Instagram usernames. """
        data_path = config.PATH_TO_INSTAGRAM_DATA
        return set(os.listdir(data_path))

    def _get_all_instagram_users(self):
        """ Returns a set of all valid instagram usernames.

        Side effect:
        1. Update the interim/ folder with new preprocessed data
        """
        pprocess_facade = PreProcessFacade("raw")
        data = pprocess_facade.process_pipeline()
        instagram_df = data["instagram_df"]
        if instagram_df.empty:
            print("Result from raw preprocessing is empty.")
            raise Exception
        return set(instagram_df["instagram_user_name"])
