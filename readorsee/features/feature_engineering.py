from readorsee import settings
from readorsee.data.models import InstagramUser
from readorsee.data.preprocessing import Tokenizer
import h5py
import os
import numpy as np
import pandas as pd
import liwc
import cv2
from typing import *
from collections import Counter

Path = str
parse, category_names = liwc.load_token_parser(settings.PATH_TO_PT_LIWC)
tokenizer = Tokenizer()

__all__ = ["get_features"]


def get_features(
    profile: InstagramUser, period: int
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    posts = profile.get_posts_from_qtnre_answer_date(period)
    faces = []
    likes = []
    captions = []
    comments = []
    hue = []
    saturation = []
    value = []
    post_features = {"likes_count": likes, "comments_count": comments}
    visual_features = {
        "faces_count": faces,
        "hue": hue,
        "saturation": saturation,
        "value": value,
    }
    for p in posts:
        faces.append(sum(p.get_face_count_list()))
        likes.append(p.likes_count)
        captions.append(p.caption)
        comments.append(p.comments_count)
        h, s, v = get_visual_features(p.get_img_path_list())
        hue.append(h)
        saturation.append(s)
        value.append(v)

    textual_features = get_textual_features(captions)
    visual_features = get_mean_std_from_ftrs(visual_features)
    post_features = get_mean_std_from_ftrs(post_features)
    # features_dict.update(textual_features)
    return post_features, visual_features, textual_features


def get_mean_std_from_ftrs(dic):
    new_dic = {}
    for name, ftrs in dic.items():
        d = {
            f"{name}_mean": np.mean(ftrs),
            f"{name}_std": np.std(ftrs),
            f"{name}_count": np.sum(ftrs),
        }
        new_dic.update(d)
    return new_dic


def get_visual_features(paths: List[Path]) -> List[float]:
    """
    Returns the mean of hue, saturation and value (HSV) for a list of images paths.
    The return value will always be of size 3.
    """
    hsv_list = []
    for path in paths:
        img_path = os.path.join(settings.PATH_TO_INSTAGRAM_DATA, path)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        avg_color_per_row = np.average(img, axis=0)
        hsv = np.average(avg_color_per_row, axis=0)
        hsv_list.append(hsv)
    hsv = np.vstack(hsv_list)
    hsv = np.mean(hsv, axis=0)
    return hsv


def get_textual_features(captions: List[str]) -> Dict[str, float]:
    tokens_list = []
    for c in captions:
        tokens = tokenizer.tokenize(c)
        tokens_list.extend(tokens)
    counts = dict(Counter(category for token in tokens_list for category in parse(token)))
    features = dict(zip(category_names, [0] * len(category_names)))
    for k, v in counts.items():
        features[k] = v
    return features
