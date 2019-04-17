import pandas as pd
import os
import numpy as np
import re
import json
from models import InstagramPost, InstagramUser, Questionnaire
import random
import time


class PreProcessingError(Exception):
    pass


class PreProcess:

    """This is a superclass abstraction to preprocess data, in order to
    effectively use it, it's necessary to extends this class and override
    the respective methods to load, preprocess and save processed data.
    """

    def __init__(self):
        pass

    def _load_data(self):
        raise NotImplementedError

    def preprocess(self):
        raise NotImplementedError

    def save_processed(self):
        raise NotImplementedError


class RawPreProcess(PreProcess):

    """ This is a helper class to manipulate the questionnaire dataframe,
    pre-processing it, producing three output dataframes:

    _out_instagram_df -- DataFrame whose participants contain valid instagram
    usernames.
    _out_twitter_df -- DataFrame whose participants contain valid twitter
    usernames.
    _out_all_participants_df -- DataFrame with all participants, even with
    invalid usernames.
    """

    def __init__(self):
        """ Initialize all dataframes used in this class. """
        super().__init__()
        self.dataframe = self._load_data()
        self._out_instagram_df = None
        self._out_twitter_df = None
        self._out_all_participants_df = None

    def preprocess(self):
        print("Preprocessing raw data...")
        """ Preprocess the questionnaire data to normalize it. """
        return self._preprocess_questionnaire(self.dataframe.copy())

    def _load_data(self):
        """ Return the questionnaire DataFrame. """
        root_path = os.path.abspath("")
        data_path = os.path.join(root_path, "..", "..",
                                 "data", "raw", "questionnaire.csv")
        df = pd.read_csv(data_path, encoding="utf-8")
        return df

    def save_processed(self):
        """ Save all three generated DataFrames after preprocessing.

        DataFrames:
        1. Instagram : Contains only valid instagram useranems
        2. Twitter   : Contains only valid Twitter usernames
        3. All Participants : Contains all participants data
        """
        self._save_dataframe(self._out_instagram_df, "instagram.csv")
        self._save_dataframe(self._out_twitter_df, "twitter.csv")
        self._save_dataframe(self._out_all_participants_df,
                             "all_participants.csv")

        data = dict(instagram_df=self._out_instagram_df,
                    twitter_df=self._out_twitter_df,
                    all_participants_df=self._out_all_participants_df)

        return data

    def _get_number(self, choice):
        pattern = re.compile(r"(\d)(\.|\w)", re.UNICODE)
        match = re.match(pattern, choice)
        return int(match.group(1))

    def _salary_normalization(self, text):
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

    def _rename_columns(self, columns):
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

    def _normalize_usernames(self, df):
        """ Put usernames to lowercase and remove @ from username. """
        df["instagram_user_name"] = (df["instagram_user_name"]
                                     .map(lambda x: str(x).lower()
                                          .replace("@", "").strip()))
        df["twitter_user_name"] = (df["twitter_user_name"]
                                   .map(lambda x: str(x).lower()
                                        .replace("@", "").strip()))
        return df

    def _remove_malformed_usernames(self, df, column):
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
        df = self._normalize_usernames(df)
        df = df.drop_duplicates(subset=column)
        return df

    def _save_dataframe(self, dataframe, filename):
        """ Save the dataframe with the filename name. """
        root_path = os.path.abspath("")
        data_path = os.path.join(root_path, "..", "..",
                                 "data", "interim", filename)
        dataframe.to_csv(data_path, encoding="utf-8", index=False)

    def _preprocess_questionnaire(self, df):
        """ Pre-process the questionnaire data and return it.

        We have Instagram and Twitter dataframes because of username
        malformation. Since it is possible for users to input malformed
        usernames while answering the questionnaire, we remove answers (rows)
        with malformed usernames.

        Return:
        instagram_df -- Instagram dataframe
        twitter_df -- Twitter dataframe
        all_participants_df -- All participants dataframe
        """
        df.iloc[:, 24:45] = df.iloc[:, 24:45].applymap(self._get_number)
        df["BDI"] = df.iloc[:, 24:45].apply(np.sum, axis=1)
        df.iloc[:, 7] = df.iloc[:, 7].transform(self._salary_normalization)

        new_columns = self._rename_columns(df.columns.values)
        df.columns = new_columns

        bdi_range = list(range(24, 45))
        drop_columns = [2, 3, 4, 8, 18, 19, 23, 47] + bdi_range

        df = df.drop(df.columns[drop_columns], axis=1)

        df["form_application_date"] = (
            pd.to_datetime(df["form_application_date"]))
        df["form_application_date"] = df["form_application_date"].apply(
            lambda x: x.date)
        df["email"] = df["email"].transform(lambda x: x.lower())

        instagram_df = df.copy()
        twitter_df = df.copy()

        instagram_df = self._remove_malformed_usernames(instagram_df,
                                                        "instagram_user_name")
        twitter_df = self._remove_malformed_usernames(twitter_df,
                                                      "twitter_user_name")

        self._out_instagram_df = instagram_df
        self._out_twitter_df = twitter_df
        self._out_all_participants_df = df

        data = dict(instagram_df=self._out_instagram_df,
                    twitter_df=self._out_twitter_df,
                    all_participants_df=self._out_all_participants_df)

        return data


class InstagramExternalPreProcess(PreProcess):

    """
    Helper class for preprocessing the external data for instagram posts. It
    opens the external instagram users data.

    Data generated:
    _out_instagram_df -- The final, canonical dataset for the instagram sample.
        The dataset contains all users that are fed to the machine learning
        models, it is as csv file, and only contains _valid_participants data.
    _valid_participants -- A list of InstagramUser models that contains only
        valid users (open profiles, and contain at least 1 post)
    _stratified_hdf_list -- A list of n stratified datasets samples
    """

    def __init__(self):
        super().__init__()
        self._out_instagram_df = None
        self._valid_participants = None
        self._stratified_hdf_list = None
        self._blocked_profiles = None

    def preprocess(self):
        """ Load and preprocess all instagram posts from all participants
        (external folder) with valid and open instagram profiles. Only users
        with at least 1 post are considered in this study.

        Return:
        valid_participants -- All valid participants (at least 1 post)
        blocked_profiles -- All blocked instagram profiles
        """
        print("Preprocessing external data...")
        base_path = os.path.abspath("")
        data_folder = os.path.join(base_path, "..", "..", "data", "external",
                                   "instagram")
        answers_df = self._load_instagram_questionnaire_answers()

        self._blocked_profiles = []
        self._valid_participants = []

        for user in self._load_data(data_folder):

            if self._has_at_least_one_post(user["json"]):
                instagram_user, instagram_posts = self._get_instagram_models(
                    os.path.join(user["folder_path"]), user["username"],
                    user["json"], answers_df)

                if (instagram_user is None) or (instagram_posts is None):
                    self._blocked_profiles.append(user["username"])
                else:
                    self._valid_participants.append(instagram_user)
            else:
                self._blocked_profiles.append(user["username"])

        self._blocked_profiles.remove("instagram")
        self._out_instagram_df = self._create_instagram_df(
                                            self._valid_participants)

        data = dict(valid_participants=self._valid_participants,
                    blocked_profiles=self._blocked_profiles,
                    instagram_df=self._out_instagram_df)

        return data

    def _load_data(self, data_folder):
        for root, dirs, files in os.walk(data_folder):
            username = os.path.basename(root)
            user_json = self._get_user_json(root, username)
            data = dict(json=user_json, username=username, folder_path=root)
            yield data

    def _load_instagram_questionnaire_answers(self):
        base_path = os.path.abspath("")
        answers_path = os.path.join(base_path, "..", "..", "data", "interim",
                                    "instagram.csv")
        return pd.read_csv(answers_path, encoding="utf-8")

    def _get_user_json(self, root, username):
        """ Return the JSON that contains the user information.

        Return:
        user_json -- JSON object with Instagram posts and user profile data
        """
        json_path = os.path.join(root, username + ".json")

        try:

            with open(json_path, "r") as f:
                user_json = f.read()
            user_json = json.loads(user_json)

        except Exception:
            user_json = None

        return user_json

    def _has_at_least_one_post(self, user_json):
        """ Check if this user profile is not blocked for public view.

        We only need to check if the "GraphImages" object exists inside the
        JSON file and then check if it has at least 1 post.
        """
        if user_json:
            graph_images = user_json.get("GraphImages", [])
            return True if len(graph_images) > 0 else False
        return False

    def _get_instagram_models(self, root_path, username, user_json,
                              answers_df):
        """ Uses the models written in models.py file to model
        instagram posts into proper classes. It assumes that the user_json
        was already checked to have the GraphImages object.

        Return:
        instagram_user -- InstagramUser model
        instagram_posts -- A list of InstagramPost classes
        None -- if there is no post objects
        """
        posts_json = user_json["GraphImages"]
        profile_json = user_json["GraphProfileInfo"]
        instagram_posts = []

        qtnre_answer = self._get_qtnre_model(username, answers_df)

        for post in posts_json:
            instagram_post = self._get_instagram_post_model(root_path, post)

            if instagram_post is None:
                continue

            instagram_posts.append(instagram_post)

        if not instagram_posts:
            return None, None

        instagram_user = self._get_instagram_user_model(username,
                                                        profile_json,
                                                        qtnre_answer,
                                                        instagram_posts)

        return instagram_user, instagram_posts

    def _get_qtnre_model(self, username, answer_df):
        mask = answer_df["instagram_user_name"] == username
        answer_dict = answer_df[mask].to_dict("records")[0]
        qtnre = Questionnaire(answer_dict)
        return qtnre

    def _get_instagram_post_model(self, root_path, post):
        """ Return the InstagramPost model for this post.

        Return:
        instagram_post -- InstagramPost instance or None if this post contains
            no images
        """

        imgs_paths = self._process_post_images(root_path, post)

        if not imgs_paths:
            return None

        likes_count = post.get("edge_media_preview_like").get("count", 0)

        def get_caption(post):
            caption = post["edge_media_to_caption"]["edges"]
            if len(caption):
                caption = caption[0].get("node").get("text", "")
            else:
                caption = ""
            return caption

        caption = get_caption(post)
        comments_count = post["edge_media_to_comment"].get("count", 0)
        timestamp = post["taken_at_timestamp"]

        post = InstagramPost(imgs_paths, caption, likes_count, timestamp,
                             comments_count)

        return post

    def _process_post_images(self, root_path, post):
        """ Return images paths for this specific post.

        For Instagram, one post can possibly have two or more pictures. Return
        paths that are found in directory.
        """
        def process_url(url):
            if not url:
                return ""
            pattern = re.compile(r"/(\d*_\d*_\d*_\w\.jpg)\?", re.UNICODE)
            match = re.search(pattern, url)
            if match is None:
                return ""
            return match.group(1)

        pic_ids = list(filter(None, map(process_url,
                                        post.get("urls", []))))
        img_path_list = []

        for pid in pic_ids:
            image_path = os.path.join(root_path, pid)
            if os.path.isfile(image_path):
                img_path_list.append(image_path)

        return img_path_list

    def _get_instagram_user_model(self, username, profile, qtnre_answer,
                                  instagram_posts):
        """ Create and return an InstagramUser model. """
        profile_info = profile["info"]
        biography = profile_info.get("biography", "")
        followers_count = profile_info.get("followers_count", 0)
        following_count = profile_info.get("following_count", 0)
        is_private = profile_info.get("is_private", True)
        posts_count = profile_info.get("posts_count", 0)
        user = InstagramUser(biography, followers_count, following_count,
                             is_private, posts_count, qtnre_answer, username,
                             instagram_posts)
        return user

    def _create_instagram_df(self, valid_participants):
        """ Valid user profiles are (1) open profiles, and (2) profiles with
        at least one post."""

        if not valid_participants:
            raise PreProcessingError

        def get_original_csv_cols_order():
            """ Get the original answers cols order to keep it normalized
            in the new dataframe. """
            qtnre_answers = self._load_instagram_questionnaire_answers()
            cols_order = qtnre_answers.columns.tolist()
            return cols_order

        cols_order = get_original_csv_cols_order()

        questionnaire_answers = []
        for profile in valid_participants:
            answer_dict, keys = profile.get_answer_dict()
            questionnaire_answers.append(answer_dict)

        self._out_instagram_df = pd.DataFrame(questionnaire_answers,
                                              columns=cols_order + keys)

        return self._out_instagram_df

    def save_processed(self):
        """ TODO: finish this function """
        base_path = os.path.abspath("")
        canonical_csv_path = os.path.join(base_path, "..", "..", "data",
                                          "processed", "instagram.csv")
        self._out_instagram_df.to_csv(canonical_csv_path, index=False)

        data = dict(participants=self._valid_participants,
                    blocked_profiles=self._blocked_profiles,
                    instagram_df=self._out_instagram_df)

        return data


class LocalSearch:

    """ TabuSearch to stratify the dataset."""

    def __init__(self, tr_size=0.6, val_size=0.2, te_size=0.2):
        self.tr_size = tr_size
        self.te_size = te_size
        self.val_size = val_size
        if tr_size + te_size + val_size != 1.0:
            raise ValueError
        self._original_bdi_qty = {0: 0, 1: 0}
        self._original_size = -1
        self._days = 60
        self._timer = None

    def stratify(self, participants, days):
        """ Return the participants list stratified through train/val/test
        size, class, and number of examples for each participant.

        We stratify by 3 levels: train/test/val size, class, and examples

        Params:
        participants -- A list of models.Participant subclasses
        Return:
        the stratified participants list
        """
        print("Stratifying...")
        self._timer = time.time()
        self._days = days
        prtcpts = np.array(self._get_participants_with_posts(participants))
        self._original_bdi_qty, self._original_size = self._calculate_bdi_qty(
            prtcpts)
        best = tuple(range(len(prtcpts)))
        best_candidate = best
        tabuList = [hash(best)]
        TABU_SIZE = int(0.3*len(prtcpts))

        while not self._stopping_condition(prtcpts, best):
            neighbors = self._get_neighbors(prtcpts, best_candidate)
            for n in neighbors:
                if (hash(n) not in tabuList
                   and self._fitness(prtcpts, n)
                   < self._fitness(prtcpts, best_candidate)):
                    best_candidate = n

            if (self._fitness(prtcpts, best_candidate)
                    < self._fitness(prtcpts, best)):
                best = best_candidate

            if random.random() < 0.30:
                best_candidate = random.choice(neighbors)

            tabuList.append(hash(best_candidate))
            if len(tabuList) > TABU_SIZE:
                tabuList.pop(0)

        return self._get_subsets(prtcpts, best)

    def _get_participants_with_posts(self, participants):
        """ Return participants that have, at least, one post in the previous
        days' from the questionnaire answer. """
        participants_with_posts = []
        for p in participants:
            posts = p.get_posts_from_qtnre_answer_date(self._days)
            if len(posts) > 0:
                participants_with_posts.append(p)

        return participants_with_posts

    def _stopping_condition(self, participants, best):

        fit_value = self._fitness(participants, best)

        epsilon = 0.08
        print("Fit Value : {}".format(fit_value))
        if fit_value <= epsilon or time.time() - self._timer >= 300:
            return True
        return False

    def _get_neighbors(self, participants, best_candidate):
        NEIGHBORS = 15

        tr_idx, val_idx = self._get_indexes(participants)
        neighbors = []

        for n in range(NEIGHBORS):
            if n % 2 == 0:
                indexes = [random.randrange(0, tr_idx),
                           random.randrange(tr_idx, val_idx),
                           random.randrange(val_idx, len(participants))]
                swap_idx = random.sample(indexes, 2)
                neighbor = list(best_candidate)
                temp = neighbor[swap_idx[0]]
                neighbor[swap_idx[0]] = neighbor[swap_idx[1]]
                neighbor[swap_idx[1]] = temp
            else:
                neighbor = random.sample(best_candidate, len(best_candidate))
            neighbors.append(tuple(neighbor))

        return neighbors

    def _fitness(self, participants, mask):
        """ Return the sum of differences of the maximum and minimum of: (1)
        proportions of examples for each BDI category, and (2) proportion of
        examples in each generated set (test, train, and validation)
        """
        tr_subset, val_subset, test_subset = self._get_subsets(participants,
                                                               mask)

        def get_bdi_fraction(bdi_qty, qty):
            return (bdi_qty[0] / qty), qty

        original_bdi_0_frac, total_qty = get_bdi_fraction(
            self._original_bdi_qty,
            self._original_size)

        original_tr_size = total_qty*self.tr_size
        original_val_size = total_qty*self.val_size
        original_test_size = total_qty*self.te_size

        tr_bdi_0_frac, tr_qty = get_bdi_fraction(
            *self._calculate_bdi_qty(tr_subset))
        val_bdi_0_frac, val_qty = get_bdi_fraction(
            *self._calculate_bdi_qty(val_subset))
        test_bdi_0_frac, test_qty = get_bdi_fraction(
            *self._calculate_bdi_qty(test_subset))

        bdi_proportions = [np.abs(tr_bdi_0_frac - original_bdi_0_frac),
                           np.abs(val_bdi_0_frac - original_bdi_0_frac),
                           np.abs(test_bdi_0_frac - original_bdi_0_frac)]

        sets_proportions = [np.abs(tr_qty - original_tr_size),
                            np.abs(val_qty - original_val_size),
                            np.abs(test_qty - original_test_size)]

        # Normalization process. Necessary due to the discrepancies between
        # bdi_proportions and sets_proportions values.
        bdi_proportions = bdi_proportions / np.linalg.norm(bdi_proportions,
                                                           ord=1)
        sets_proportions = sets_proportions / np.linalg.norm(sets_proportions,
                                                             ord=1)

        # Here, we use the difference between the max and the min value to
        # weight the generated solutions. More discrepancies in the generated
        # set, more weighted they become.
        return ((np.max(bdi_proportions) - np.min(bdi_proportions))
                + (np.max(sets_proportions) - np.min(sets_proportions)))

    def _get_subsets(self, participants, mask):
        tr_idx, val_idx = self._get_indexes(participants)
        chosen_sets = participants[list(mask)]
        tr_subset = chosen_sets[:tr_idx]
        val_subset = chosen_sets[tr_idx:val_idx]
        test_subset = chosen_sets[val_idx:]
        return tr_subset, val_subset, test_subset

    def _get_indexes(self, participants):
        """ Return the maximum index for training and val sets.

        It's not necessary to return training index since it's the last element
        in the array. Typically:
        training set = [00% - 60%)
        val set      = [60% - 80%)
        training set = [80% - 100%]
        """
        tr_idx = int(np.floor(self.tr_size*len(participants)))
        j = self.val_size + self.tr_size
        val_idx = int(np.floor(j*len(participants)))
        return tr_idx, val_idx

    def _calculate_bdi_qty(self, subset):
        bdi_fraction = {0: 0, 1: 0}
        for participant in subset:
            posts = participant.get_posts_from_qtnre_answer_date(self._days)
            qty = len(posts)
            bdi = participant.questionnaire.get_binary_bdi()
            bdi_fraction[bdi] += qty

        return bdi_fraction, (bdi_fraction[0] + bdi_fraction[1])


class PreProcessCOR:

    """ Chain-of-responsibility pattern class to abstract the idea of processing
    various preprocessing steps, in order.

    """

    def __init__(self, pre_processors=None):
        """ Receives the preprocessors, needs to be ordered in a way that each
        steps precedes the next in the preprocessing pipeline.

        Params:
            pre_processors -- A list of PreProcess subclass instances
        """
        self._pre_processors = list()
        if pre_processors is not None:
            self._pre_processors += pre_processors

    def process_pipeline(self, save):
        """ Preprocess pipeline and return the last processed data, if any. """
        final_dataset = None
        for pre_process in self._pre_processors:
            pre_process.preprocess()
            if save:
                final_dataset = pre_process.save_processed()
        return final_dataset


def preprocess_pipeline(process_method="raw", save=True):
    """ Preprocess pipeline and return the last generated dataset in
    the pipeline

    Params:
        process_method -- 'complete' for the entire preprocessing pipeline,
                          'raw' to only preprocess the raw data
                          'external' to only preprocess the external data
        save -- True if you want to save the processed data
                False otherwise

    Return:
        data -- if the process is successful, the data will be returned.
                if save = False, then no data will be returned.
    """
    if process_method == "complete":
        pipeline = [RawPreProcess(), InstagramExternalPreProcess()]
    elif process_method == "raw":
        pipeline = [RawPreProcess()]
    elif process_method == "external":
        pipeline = [InstagramExternalPreProcess()]
    else:
        print("Invalid pipeline name.")
        return None

    ppf = PreProcessCOR(pre_processors=pipeline)
    final_data = ppf.process_pipeline(save)
    return final_data
