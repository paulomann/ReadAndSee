import pandas as pd
import os
import numpy as np
import re
from models import InstagramPost, InstagramProfile


class PreProcess:

    """This is a superclass abstraction to preprocess data, in order to
    effectively use it, it's necessary to extends this class and override
    the respective methods to load, preprocess and save processed data.
    """

    def __init__(self):
        pass

    def load_data(self):
        pass

    def preprocess(self):
        pass

    def save_processed(self):
        pass


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
        self.dataframe = self.load_data()
        self._out_instagram_df = None
        self._out_twitter_df = None
        self._out_all_participants_df = None

    def preprocess(self):
        """ Preprocess the questionnaire data to normalize it. """
        return self._preprocess_questionnaire(self.dataframe.copy())

    def load_data(self):
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
        """
        df.iloc[:, 24:45] = df.iloc[:, 24:45].applymap(self._get_number)
        df["BDI"] = df.iloc[:, 24:45].apply(np.sum, axis=1)
        df.iloc[:, 7] = df.iloc[:, 7].transform(self._salary_normalization)

        new_columns = self._rename_columns(df.columns.values)
        df.columns = new_columns

        bdi_range = list(range(24, 45))
        drop_columns = [2, 3, 4, 8, 18, 19, 23, 47] + bdi_range

        df = df.drop(df.columns[drop_columns], axis=1)

        df["form_application_date"] = (pd.to_datetime(df["form_application_date"]))
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

        return instagram_df, twitter_df


class InstagramExternalPreProcess(PreProcess):

    """
    Helper class for preprocessing the external data for instagram posts. It
    opens the external instagram users data.

    Return:
    _out_instagram_df -- The final, canonical dataset for the instagram sample.
        The dataset contains all users that are fed to the machine learning
        models.
    _stratified_hdf_list -- A list of n stratified datasets samples
    """

    def __init__(self):
        super().__init__()
        self._out_instagram_df = None
        self._stratified_hdf_list = []

    def load_data(self):
        root_path = os.path.abspath("")
        all_users = os.path.join(root_path, "..", "..", "data", "external",
                                 "instagram")
        for root, dirs, files in os.walk(all_users):
            


    def preprocess(self):
        pass

    def save_processed(self):
        pass
