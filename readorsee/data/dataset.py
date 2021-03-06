import os
import pickle
from collections import defaultdict
from typing import *

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision
from allennlp.modules.elmo import batch_to_ids
from gensim.models.fasttext import load_facebook_model
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import BertTokenizer
import ftfy

from readorsee import settings
from readorsee.data.facade import StratifyFacade
from readorsee.data.models import Config
from readorsee.data.preprocessing import NLTKTokenizer, Tokenizer
from readorsee.features.feature_engineering import get_features, get_features_from_post
from readorsee.features.sentence_embeddings import PMEAN, SIF

_all_ = ["DepressionCorpus", "DepressionCorpusXLM"]


class DepressionCorpus(torch.utils.data.Dataset):
    def __init__(
        self, observation_period, dataset, subset, config, fasttext=None, transform=None
    ):
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
        self.config = config
        text_embedder = ""
        data_type = self.config.general["media_type"]
        media_config = getattr(self.config, data_type)

        if data_type in ["txt", "both"]:
            text_embedder = media_config["txt_embedder"].lower()
            if text_embedder == "fasttext":
                if fasttext is None:
                    raise ValueError
                else:
                    self.fasttext = fasttext
            elif text_embedder not in ["elmo", "bow"]:
                raise ValueError(f"{text_embedder} is not a valid embedder")

        if data_type not in ["img", "txt", "both", "ftrs"]:
            raise ValueError

        if data_type in ["img"] and text_embedder in ["elmo", "fasttext", "bow"]:
            raise ValueError("Do not use text_embedder with image only dset.")

        if transform is None:
            transform = transforms.Compose(
                [
                    transforms.Resize([224, 224], interpolation=Image.LANCZOS),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
        self._transform = transform
        subset_to_index = {"train": 0, "val": 1, "test": 2}
        subset_idx = subset_to_index[subset]
        self.text_embedder = text_embedder
        self._data_type = data_type
        self._subset = subset
        self._dataset = dataset
        self._ob_period = int(observation_period)
        self._tokenizer = NLTKTokenizer()
        # A list of datasets which in turn are a list
        self._raw = StratifyFacade().load_stratified_data()
        self._raw = self._raw["data_" + str(self._ob_period)][self._dataset]
        self._raw = self._raw[subset_idx]
        self._data = self._get_posts_list_from_users(self._raw)
        self._data = self.slice_if_rest_one(self._data)
        self._users_df = pd.DataFrame()
        self._posts_df = pd.DataFrame()

        if data_type in ["txt", "both"]:
            if text_embedder == "elmo":
                self._elmo = self.preprocess_elmo()
            elif text_embedder == "fasttext":
                self._fasttext = self.preprocess_fasttext()
            elif text_embedder == "bow":
                self._bow = self.preprocess_bow()
        elif data_type == "ftrs":
            self._ftrs = self.slice_ftrs(self._get_features())

    def slice_if_rest_one(self, data):
        if self.config.general["media_type"] == "ftrs":
            return data
        size = len(data)
        bs = self.config.general["batch_size"]
        bs = bs / len(self.config.general["gpus"])
        if size % bs == 1:
            return data[:-1]
        return data

    def slice_ftrs(self, ftrs):
        size = ftrs.size(0)
        bs = self.config.general["batch_size"]
        bs = bs / len(self.config.general["gpus"])
        if size % bs == 1:
            return ftrs[:-1, ...]
        return ftrs

    @property
    def raw(self):
        return self._raw

    @property
    def users_df(self):
        return self._users_df

    @property
    def posts_df(self):
        return self._posts_df

    def get_users_features_names(self):
        return self.users_df.columns.levels[1]

    def get_posts_features_names(self):
        return self.posts_df.columns.levels[1]

    def __len__(self):
        if self._data_type == "ftrs":
            return self._ftrs.shape[0]
        return len(self._data)

    def __getitem__(self, idx):
        """ Returns a 4-tuple with img, caption, label, u_name """
        img, caption, label, u_name = self._data[idx]

        if self._data_type == "ftrs":
            u_name = self._users_df.loc[idx, ("questionnaire_ftrs", "id")]
            label = self._users_df.loc[idx, ("questionnaire_ftrs", "label")]
            if self._subset in ["train", "val"]:
                return (self._ftrs[idx], label)
            else:
                return (self._ftrs[idx], label, u_name)

        if self.text_embedder == "elmo":
            caption = (self._elmo[idx],)
        elif self.text_embedder == "fasttext":
            caption = self._fasttext[idx]
        elif self.text_embedder == "bow":
            caption = (self._bow[idx],)

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
                images_paths = [
                    os.path.join(settings.PATH_TO_INSTAGRAM_DATA, p)
                    for p in post.get_img_path_list()
                ]
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

        embeddings = []
        for txt in texts:
            text = np.array([self.fasttext.wv[token] for token in txt])
            embeddings.append(torch.from_numpy(text))

        max_size = np.max([e.size(0) for e in embeddings])
        masks = [
            torch.cat([torch.ones(e.size(0)), torch.zeros(max_size - e.size(0))])
            for e in embeddings
        ]
        masks = torch.stack(masks, dim=0)

        embeddings = torch.stack(
            [
                torch.cat([e, torch.zeros((max_size - e.size(0), 300))], 0)
                for e in embeddings
            ],
            dim=0,
        )

        return list(zip(embeddings, masks))

    def preprocess_bow(self):
        _, texts, _, _ = zip(*self._data)
        corpus = [" ".join(tokens) for tokens in texts]
        if self._subset == "train":
            tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2)
            corpus = tfidf_vectorizer.fit_transform(corpus)
            with open(settings.PATH_TO_SERIALIZED_TFIDF, "wb") as f:
                pickle.dump(tfidf_vectorizer, f)
        else:
            with open(settings.PATH_TO_SERIALIZED_TFIDF, "rb") as f:
                tfidf_vectorizer = pickle.load(f)
            corpus = tfidf_vectorizer.transform(corpus)
        corpus = torch.from_numpy(corpus.toarray()).float()
        self.bow_ftrs_size = corpus.size(1)
        return corpus

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
                d["BDI"] = u.questionnaire.get_bdi(category=False)
                ftrs = get_features_from_post(post)
                d.update(ftrs)
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
            post_ftrs_list = []
            txt_ftrs_list = []
            vis_ftrs_list = []
            for profile in participants:
                answer_dict, keys = profile.get_answer_dict()
                answer_dict["BDI"] = profile.questionnaire.get_bdi(category=False)
                # keys.append("BDI")
                post_ftrs, vis_ftrs, txt_ftrs = get_features(profile, self._ob_period)
                self.swap_features(answer_dict, post_ftrs)
                post_ftrs_list.append(post_ftrs)
                txt_ftrs_list.append(txt_ftrs)
                vis_ftrs_list.append(vis_ftrs)
                questionnaire_features.append(answer_dict)
            post_ftrs = pd.DataFrame(post_ftrs_list)
            txt_ftrs = pd.DataFrame(txt_ftrs_list)
            vis_ftrs = pd.DataFrame(vis_ftrs_list)
            questionnaire_features = pd.DataFrame(
                questionnaire_features, columns=cols_order + keys
            )
            questionnaire_features.drop(
                ["following_count", "followers_count"], inplace=True, axis=1
            )
            questionnaire_features = self.delete_rename_and_categorize_cols(
                questionnaire_features
            )
            return pd.concat(
                [questionnaire_features, post_ftrs, vis_ftrs, txt_ftrs],
                keys=["questionnaire_ftrs", "post_ftrs", "vis_ftrs", "txt_ftrs"],
                axis=1,
            )

        return get_answers_df(subset)

    def _load_instagram_questionnaire_answers(self):
        answers_path = os.path.join(settings.PATH_TO_INTERIM_DATA, "instagram.csv")
        return pd.read_csv(answers_path, encoding="utf-8")

    def delete_rename_and_categorize_cols(self, df):
        remove_columns = [
            "email",
            "course_name",
            "form_application_date",
            "birth_date",
            "course_name",
            "twitter_user_name",
            "accommodation",
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
            "antidepressants",
        ]
        for col in convert_cols:
            new_df[col] = pd.factorize(new_df[col], sort=True)[0]
        new_df.rename(
            {"instagram_user_name": "id", "binary_bdi": "label"}, axis=1, inplace=True
        )
        return new_df

    def swap_features(self, bio_ftrs, data_ftrs):
        data_ftrs["following_count"] = bio_ftrs["following_count"]
        data_ftrs["followers_count"] = bio_ftrs["followers_count"]
        del bio_ftrs["following_count"]
        del bio_ftrs["followers_count"]

    def get_normalized_df(self) -> pd.DataFrame:
        def remove_cols(df):
            df = df.drop([("questionnaire_ftrs", "id")], axis=1)
            df = df.drop([("questionnaire_ftrs", "label")], axis=1)
            df = df.drop([("questionnaire_ftrs", "BDI")], axis=1)
            return df

        params = getattr(self.config, self.config.general["media_type"])
        users_df = self.get_participants_dataframes()
        users_df = remove_cols(users_df)
        if params["features"] == "vis_ftrs" or params["features"] == "txt_ftrs":
            users_df = users_df[[params["features"], "post_ftrs"]]
        elif params["features"] == "both":
            users_df = users_df[["txt_ftrs", "vis_ftrs", "post_ftrs"]]
        elif params["features"] == "form_ftrs":
            return users_df
        else:
            raise ValueError("Incorrect feature type value.")
        return users_df

    def _get_features(self) -> torch.Tensor:
        users_df = self.get_normalized_df()
        X = torch.from_numpy(users_df.to_numpy()).float()
        return X


class DepressionCorpusTransformer(torch.utils.data.Dataset):
    def __init__(self, observation_period, dataset, subset, config, transform=None):
        self.config = config
        text_embedder = ""
        data_type = self.config.general["media_type"]
        media_config = getattr(self.config, data_type)
        text_embedder = media_config["txt_embedder"].lower()
        if text_embedder not in ["xlm"]:
            raise ValueError(
                f"{text_embedder} must be the 'xlm' embedder to be using DepressionCorpusXLM dataset."
            )

        if data_type not in ["txt", "both"]:
            raise ValueError(
                f"Data type '{data_type}' is not valid. It must be one of ['txt', 'both']"
            )
        if transform is None:
            transform = transforms.Compose(
                [
                    transforms.Resize([224, 224], interpolation=Image.LANCZOS),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
        self._transform = transform
        subset_to_index = {"train": 0, "val": 1, "test": 2}
        subset_idx = subset_to_index[subset]
        self.text_embedder = text_embedder
        self._data_type = data_type
        self._subset = subset
        self._dataset = dataset
        self._ob_period = int(observation_period)
        self._tokenizer = self._initialize_tokenizer()
        # A list of datasets which in turn are a list
        self._raw = StratifyFacade().load_stratified_data()
        self._raw = self._raw["data_" + str(self._ob_period)][self._dataset]
        self._raw = self._raw[subset_idx]
        self._data = self._get_posts_list_from_users(self._raw)

    def _initialize_tokenizer(self) -> BertTokenizer:
        class_model = self.config.general["class_model"].lower()
        bert_size = self.config.general["bert_size"].lower()
        if "bert" not in class_model:
            raise ValueError(
                f"The parameter 'class_model' should be one of the BERT models, not {class_model}."
            )
        if bert_size not in ["large", "base"]:
            raise ValueError(
                f"The parameter 'bert_size' should be 'large' or 'base', not {bert_size}"
            )

        return BertTokenizer.from_pretrained(settings.PATH_TO_BERT[bert_size])

    def _get_posts_list_from_users(self, user_list):
        """ Return a list of posts from a user_list
        
        This function consider an instagram post with multiples images as 
        multiples posts with the same caption for all images in the same post.
        """
        data = []
        for u in user_list:
            for post in u.get_posts_from_qtnre_answer_date(self._ob_period):
                images_paths = [
                    os.path.join(settings.PATH_TO_INSTAGRAM_DATA, p)
                    for p in post.get_img_path_list()
                ]
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
        # Token indices sequence length is longer than the specified maximum sequence
        # length for this model (530 > 512). Running this sequence through the model
        # will result in indexing errors
        # text = self._tokenizer.encode(text, max_length=511, return_tensors="pt")
        # text = self._tokenizer.tokenize(ftfy.fix_text(text))
        text = self._tokenizer.encode_plus(
            ftfy.fix_text(text),
            add_special_tokens=True,
            max_length=150,
            pad_to_max_length=True,
            return_tensors="pt",
            return_attention_mask=True,
        )
        if self._data_type in ["img", "both"]:
            image = Image.open(img_path)
            img = image.copy()
            image.close()
            if self._transform is not None:
                img = self._transform(img)
        else:
            img = img_path

        return img, text

    def __len__(self):
        return len(self._data)

    def __getitem__(self, i):
        img, caption, label, u_name = self._data[i]

        if self._data_type == "txt":
            data = (caption,)
        elif self._data_type == "both":
            data = (img,) + caption

        if self._subset in ["train", "val"]:
            return data + (label,)
        return data + (label, u_name)


class IterableDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        observation_period,
        data_type,
        dataset,
        subset,
        transform=None,
        preprocess_text=True,
        preprocess_img=True
    ):
        if data_type not in ["img", "txt", "both"]:
            raise ValueError(
                f"Data type '{data_type}' is not valid. It must be one of ['txt', 'both']"
            )
        if transform is None:
            transform = transforms.Compose(
                [
                    transforms.Resize([224, 224], interpolation=Image.LANCZOS),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
        self.preprocess_text = preprocess_text
        self.preprocess_img = preprocess_img
        self._transform = transform
        subset_to_index = {"train": 0, "val": 1, "test": 2}
        subset_idx = subset_to_index[subset]
        self._data_type = data_type
        self._subset = subset
        self._dataset = dataset
        self._ob_period = int(observation_period)
        self._tokenizer = NLTKTokenizer()
        # A list of datasets which in turn are a list
        self._raw = StratifyFacade().load_stratified_data()
        self._raw = self._raw["data_" + str(self._ob_period)][self._dataset]
        self._raw = self._raw[subset_idx]
        self._data = self._get_posts_list_from_users(self._raw)

    def _get_posts_list_from_users(self, user_list):
        """ Return a list of posts from a user_list
        
        This function consider an instagram post with multiples images as 
        multiples posts with the same caption for all images in the same post.
        """
        data = []
        for u in user_list:
            for post in u.get_posts_from_qtnre_answer_date(self._ob_period):
                images_paths = [
                    os.path.join(settings.PATH_TO_INSTAGRAM_DATA, p)
                    for p in post.get_img_path_list()
                ]
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
        # Token indices sequence length is longer than the specified maximum sequence
        # length for this model (530 > 512). Running this sequence through the model
        # will result in indexing errors
        # text = self._tokenizer.encode(text, max_length=511, return_tensors="pt")
        text = (
            self._tokenizer.tokenize(ftfy.fix_text(text))[:511]
            if self.preprocess_text
            else text
        )
        if self._data_type in ["img", "both"]:
            image = Image.open(img_path)
            img = image.copy()
            image.close()
            if self._transform is not None:
                img = self._transform(img) if self.preprocess_img else img
        else:
            img = img_path

        return img, text

    def __len__(self):
        return len(self._data)

    def __getitem__(self, i):
        img, caption, label, u_name = self._data[i]

        if self._data_type == "txt":
            data = (caption,)
        elif self._data_type == "img":
            data = (img,)
        elif self._data_type == "both":
            data = (img,) + (caption)

        if self._subset in ["train", "val"]:
            return data + (label,)
        return data + (label, u_name)



class TransferDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        observation_period,
        dataset,
        subset,
    ):
        subset_to_index = {"train": 0, "val": 1, "test": 2}
        subset_idx = subset_to_index[subset]
        self._subset = subset
        self._dataset = dataset
        self._ob_period = int(observation_period)
        # A list of datasets which in turn are a list
        self._raw = StratifyFacade().load_stratified_data()
        self._raw = self._raw["data_" + str(self._ob_period)][self._dataset]
        self._raw = self._raw[subset_idx]
        self._data = self._get_posts_list_from_users(self._raw)

    def _get_posts_list_from_users(self, user_list):
        """ Return a list of posts from a user_list
        
        This function consider an instagram post with multiples images as 
        multiples posts with the same caption for all images in the same post.
        """
        data = []
        for u in user_list:
            for post in u.get_posts_from_qtnre_answer_date(self._ob_period):
                images_paths = [p for p in post.get_img_path_list()]
                text = post.caption
                label = u.questionnaire.get_bdi(category=False)
                u_name = u.username
                data.append((images_paths, text, label, u_name))

        return data

    def __len__(self):
        return len(self._data)

    def __getitem__(self, i):
        img, caption, label, u_name = self._data[i]
        return (img, caption, label, u_name)
        # print(self._data[i])
        # data = (img,) + (caption)
        # return data + (label, u_name)
