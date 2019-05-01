import os
from dotenv import load_dotenv, find_dotenv
import pickle
import pandas as pd
import numpy as np
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


class DataLoader:

    def __init__(self):
        self._raw = None
        self._user_dfs = {}
        self._posts_dfs = {}

    @property
    def raw(self):
        return self._raw

    @property
    def user_dfs(self):
        return self._user_dfs

    @property
    def posts_dfs(self):
        return self._posts_dfs

    def get_most_similar_datasets(self, tr_frac=0.6, val_frac=0.2,
                                  test_frac=0.2):
        """ Return a dict containing the index of the best dataset for every
        obs. period considered.

        Return:
            A dict like {"data_<obs.period>": index}
        """
        bests = {}
        self._raw = self._raw if self._raw else self._load_raw_data()

        for key, value in self._raw.items():
            days = int(key.split("_")[1])
            dataset = self._raw[key][0]
            bests[key] = None
            original_ds = np.concatenate([dataset[0], dataset[1], dataset[2]])
            original_bdi_0_frac, original_bdi_1_frac, total_qty = (
                    self._calculate_bdi_qty(original_ds, days))

            minimum = [100000000, -1]
            for i, dataset in enumerate(self._raw[key]):
                train_bdi_0_frac, _, tr_qty = (
                    self._calculate_bdi_qty(dataset[0], days))
                val_bdi_0_frac, _, val_qty = (
                    self._calculate_bdi_qty(dataset[1], days))
                test_bdi_0_frac, _, test_qty = (
                    self._calculate_bdi_qty(dataset[2], days))

                bdis_fracs = [np.abs(train_bdi_0_frac - original_bdi_0_frac),
                              np.abs(test_bdi_0_frac - original_bdi_0_frac),
                              np.abs(val_bdi_0_frac - original_bdi_0_frac)]
                delta = sum(bdis_fracs)
                qty_fracs = [np.abs(tr_frac - tr_qty/total_qty),
                             np.abs(val_frac - val_qty/total_qty),
                             np.abs(test_frac - test_qty/total_qty)]
                delta += sum(qty_fracs)

                if delta < minimum[0]:
                    minimum[0] = delta
                    minimum[1] = i
            bests[key] = minimum[1]
        return bests

    def _calculate_bdi_qty(self, subset, days):

        def get_total_number_of_images(posts):
            total = 0
            for p in posts:
                total += len(p.get_img_path_list())
            return total

        bdi_fraction = {0: 0, 1: 0}
        for participant in subset:
            posts = participant.get_posts_from_qtnre_answer_date(days)
            # qty = len(posts)
            qty = get_total_number_of_images(posts)
            bdi = participant.questionnaire.get_binary_bdi()
            bdi_fraction[bdi] += qty
        total = (bdi_fraction[0] + bdi_fraction[1])
        return bdi_fraction[0]/total, bdi_fraction[1]/total, total

    def get_posts_dataframes(self):
        self.load_raw_data()

        self._posts_dfs = {}
        for k, v in self._raw.items():
            self._posts_dfs[k] = []
            for dset in self._raw[k]:
                train = self.get_user_posts_dfs(dset[0])
                val = self.get_user_posts_dfs(dset[1])
                test = self.get_user_posts_dfs(dset[2])
                final_df = pd.concat([train, val, test],
                                     keys=["train", "val", "test"])
                self._posts_dfs[k].append(final_df)
        return self._posts_dfs

    def get_user_posts_dfs(self, user_list):
        posts_dicts = []
        for u in user_list:
            for post in u.posts:
                d = post.get_dict_representation()
                d["instagram_username"] = u.username
                d["binary_bdi"] = u.questionnaire.get_binary_bdi()
                d["BDI"] = u.questionnaire.get_bdi(False)
                posts_dicts.append(d)
        return pd.DataFrame(posts_dicts)

    def load_raw_data(self):
        self._raw = StratifyFacade("").load_stratified_data()
        return self._raw

    def get_participants_dataframes(self):
        """ Return 10 dataframes for each period considered
        For the case of this study, we use three periods: 60, 212 and 365 days.

        Return:
        dataframes -- A dictionary whose keys are "data_<days>". The value for
            each key contains a list of K generated datasets, which for this
            study are 10.
        """
        self.load_raw_data()

        self._user_dfs = {}
        for key, value in self._raw.items():
            self._user_dfs[key] = []
            for dataset in self._raw[key]:
                train = dataset[0]
                val = dataset[1]
                test = dataset[2]
                df = self._create_instagram_user_df(train, val, test)
                self._user_dfs[key].append(df)
        return self._user_dfs

    def _create_instagram_user_df(self, train, val, test):
        """ Valid user profiles are (1) open profiles, and (2) profiles with
        at least one post."""

        def get_original_csv_cols_order():
            """ Get the original answers cols order to keep it normalized
            in the new dataframe. """
            qtnre_answers = self._load_instagram_questionnaire_answers()
            cols_order = qtnre_answers.columns.tolist()
            return cols_order

        cols_order = get_original_csv_cols_order()

        def get_answers_df(participants):
            questionnaire_answers = []
            for profile in participants:
                answer_dict, keys = profile.get_answer_dict()
                questionnaire_answers.append(answer_dict)

            df = pd.DataFrame(questionnaire_answers, columns=cols_order + keys)
            return df

        train = get_answers_df(train)
        val = get_answers_df(val)
        test = get_answers_df(test)

        final_df = pd.concat([train, val, test], keys=["train", "val", "test"])

        return final_df

    def _load_instagram_questionnaire_answers(self):
        answers_path = os.path.join(config.PATH_TO_INTERIM_DATA,
                                    "instagram.csv")
        return pd.read_csv(answers_path, encoding="utf-8")
