from readorsee.data.facade import StratifyFacade
from readorsee import settings
from readorsee.data.models import Config
import torch
from torchvision import transforms
import torchvision
import pandas as pd
import numpy as np
from PIL import Image
import os
from readorsee.data.preprocessing import Tokenizer
from readorsee.data.models import Config
from gensim.models.fasttext import load_facebook_model
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from allennlp.modules.elmo import batch_to_ids
from collections import defaultdict
from readorsee.features.sentence_embeddings import SIF, PMEAN
from readorsee.features import get_features

_all_ = ["DepressionCorpus"]


class DepressionCorpus(torch.utils.data.Dataset):

    def __init__(self, observation_period, dataset,
                 subset, fasttext=None, transform=None):
        """
        Params:
        subset: Can take three possible values: (train, test, val)
        observation_period: number of days for the period
        transform: the transformation method for images
        data_type: The type of training, with "img", "txt", or "both"
        text_embedder: ["fasttext", "elmo"]

        Observation: The best datasets for each period are :
            {'data_60': 1, 'data_212': 1, 'data_365': 5}
        """
        self.config = Config()
        text_embedder = ""
        data_type = self.config.general["media_type"]
        if data_type in ["txt", "both"]:
            text_embedder = self.config.txt["embedder"].lower()
        
        if data_type not in ["img", "txt", "both"]:
            raise ValueError
        
        if data_type in ["img"] and text_embedder in ["elmo", "fasttext"]:
            raise ValueError("Do not use text_embedder with image only dset.")

        if transform is None:
            transform = transforms.Compose(
                [transforms.Resize([224, 224], interpolation=Image.LANCZOS),
                 transforms.ToTensor(),
                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])])
        self._transform = transform

        if data_type in ["txt", "both"]:
            if text_embedder == "fasttext":
                if fasttext is None:
                    raise ValueError
                else:
                    self.fasttext = fasttext
            elif text_embedder != "elmo":
                raise ValueError(f"{text_embedder} is not a valid embedder")

        subset_to_index = {"train": 0, "val": 1, "test": 2}
        subset_idx = subset_to_index[subset]
        self.text_embedder = text_embedder
        self._data_type = data_type
        self._subset = subset
        self._dataset = dataset
        self._ob_period = int(observation_period)
        self._tokenizer = Tokenizer()
        # A list of datasets which in turn are a list
        self._raw = StratifyFacade().load_stratified_data()
        self._raw = self._raw["data_" + str(self._ob_period)][self._dataset]
        self._raw = self._raw[subset_idx]
        self._data = self._get_posts_list_from_users(self._raw)

        _, sentences, _, _ = zip(*self._data)
        self.sif_weights = SIF.get_SIF_weights(sentences)

        if data_type in ["txt", "both"] :
            if text_embedder == "elmo":
                self._elmo = self.preprocess_elmo()
            elif text_embedder == "fasttext":
                self._fasttext = self.preprocess_fasttext()

        self._users_df = pd.DataFrame()
        self._posts_df = pd.DataFrame()

    @property
    def raw(self):
        return self._raw

    @property
    def users_df(self):
        return self._users_df

    @property
    def posts_df(self):
        return self._posts_df

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        """ Returns a 4-tuple with img, caption, label, u_name """
        img, caption, label, u_name = self._data[idx]
        sif_weight = self.sif_weights[idx]

        if self.text_embedder == "elmo":
            if self.config.txt["mean"] == "sif":
                caption = (self._elmo[idx], sif_weight)
            else:
                caption = (self._elmo[idx],)
        elif self.text_embedder == "fasttext":
            caption = (self._fasttext[idx],)

        if self._data_type == "txt":
            data = caption
        elif self._data_type == "both":
            data = (img,) + caption
        elif self._data_type == "img":
            data = (img,)

        if self._subset in ["train", "val"]:
            return data + (label,)
        return data + (label, u_name)

    def _get_posts_list_from_users(self, user_list):
        """ Return a list of posts from a user_list
        
        This function consider an instagram post with multiples images as 
        multiples posts with the same caption for all images in the same post.
        """
        data = []
        for u in user_list:
            for post in u.get_posts_from_qtnre_answer_date(self._ob_period):
                images_paths = [os.path.join(settings.PATH_TO_INSTAGRAM_DATA, p)
                                for p in post.get_img_path_list()]
                if self._data_type in ["both", "txt"]:
                    images_paths = [images_paths[0]]
                text = post.caption
                label = u.questionnaire.get_binary_bdi()
                u_name = u.username
                for img_path in images_paths:
                    img, txt = self.preprocess_data(img_path, text)
                    data.append((img, txt, label, u_name))

        return data        

    def preprocess_data(self, img_path, text):
        text = self._tokenizer.tokenize(text)[:100]
        text = [""] if not text else text

        if self._data_type in ["img", "both"]:
            image = Image.open(img_path)
            img = image.copy()
            image.close()
            if self._transform is not None:
                img = self._transform(img)
        else:
            img = img_path

        return img, text

    def preprocess_elmo(self):
        _, text, _, _ = zip(*self._data)
        return batch_to_ids(text)
    
    def preprocess_fasttext(self):
        _, texts, _, _ = zip(*self._data)

        def get_mean(x, masks):

            if self.config.txt["mean"] == "sif":
                sif = SIF()
                sif_embeddings = sif.SIF_embedding(x, masks, self.sif_weights)
                return sif_embeddings

            elif self.config.txt["mean"] == "pmean":
                pmean = PMEAN()
                means = self.config.txt["pmean"]
                pmean_embedding = pmean.PMEAN_embedding(x, masks, means)
                return pmean_embedding

            elif self.config.txt["mean"] == "avg":
                x = x.sum(dim=1)
                masks = masks.sum(dim=1).view(-1, 1).float()
                x = torch.div(x, masks)
                x[torch.isnan(x)] = 0
                x[torch.isinf(x)] = 1
                return x
            else:
                raise NotImplementedError

        embeddings = []
        for txt in texts:
            text = np.array([self.fasttext.wv[token] for token in txt])
            embeddings.append(torch.from_numpy(text))
    
        max_size = np.max([e.size(0) for e in embeddings])
        masks = [torch.cat([torch.ones(e.size(0)), 
                 torch.zeros(max_size-e.size(0))]) for e in embeddings]
        masks = torch.stack(masks, dim=0)
                 
        embeddings = torch.stack([
                     torch.cat([e, torch.zeros((max_size - e.size(0), 300))], 0) 
                                for e in embeddings], dim=0)

        embeddings = get_mean(embeddings, masks)
        return embeddings

    def get_posts_dataframes(self):
        self._posts_df = self._get_users_posts_dfs(self._raw)
        return self._posts_df

    def _get_users_posts_dfs(self, user_list):
        posts_dicts = []
        for u in user_list:
            for post in u.get_posts_from_qtnre_answer_date(self._ob_period):
                d = post.get_dict_representation()
                d["instagram_username"] = u.username
                d["binary_bdi"] = u.questionnaire.get_binary_bdi()
                d["BDI"] = u.questionnaire.get_bdi(False)
                posts_dicts.append(d)
        return pd.DataFrame(posts_dicts)

    def get_participants_dataframes(self):
        self._users_df = self._create_instagram_user_df(self._raw)
        return self._users_df

    def _create_instagram_user_df(self, subset):
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
            questionnaire_features = []
            data_features = []
            for profile in participants:
                answer_dict, keys = profile.get_answer_dict()
                data_dict = get_features(profile, self._ob_period)
                self.swap_features(answer_dict, data_dict)
                data_features.append(data_dict)
                questionnaire_features.append(answer_dict)
            data_features = pd.DataFrame(data_features)
            questionnaire_features = pd.DataFrame(
                questionnaire_features, columns=cols_order + keys
            )
            questionnaire_features.drop(
                ["following_count", "followers_count"],
                inplace=True,
                axis=1
            )
            questionnaire_features = self.delete_rename_and_categorize_cols(
                questionnaire_features
            )
            return pd.concat(
                [questionnaire_features, data_features],
                keys=["questionnaire_ftrs", "data_only_ftrs"],
                axis=1
            )

        return get_answers_df(subset)

    def _load_instagram_questionnaire_answers(self):
        answers_path = os.path.join(settings.PATH_TO_INTERIM_DATA,
                                    "instagram.csv")
        return pd.read_csv(answers_path, encoding="utf-8")

    def delete_rename_and_categorize_cols(self, df):
        remove_columns = [
            "email",
            "course_name",
            "form_application_date",
            "birth_date",
            "course_name",
            "twitter_user_name",
            "accommodation"
        ]
        new_df = df.drop(remove_columns, axis=1)
        convert_cols = [
            "sex",
            "household_income",
            "academic_degree",
            "scholarship",
            "works",
            "depression_diagnosed",
            "in_therapy",
            "antidepressants"
        ]
        for col in convert_cols:
            new_df[col] = pd.factorize(new_df[col])[0]
        new_df.rename(
            {"instagram_user_name": "id", "binary_bdi": "label"}, 
            axis=1,
            inplace=True
        )
        return new_df
    
    def swap_features(self, bio_ftrs, data_ftrs):
        data_ftrs["following_count"] = bio_ftrs["following_count"]
        data_ftrs["followers_count"] = bio_ftrs["followers_count"]
        del bio_ftrs["following_count"]
        del bio_ftrs["followers_count"]