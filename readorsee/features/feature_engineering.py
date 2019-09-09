from readorsee import settings
from readorsee.data.models import InstagramUser, InstagramPost
from readorsee.data.preprocessing import Tokenizer
import h5py
import os
import numpy as np
import pandas as pd
import liwc
from skimage import io, color
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
    post_features = {"likes": likes, "comments": comments}
    visual_features = {
        "faces": faces,
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
        img = io.imread(img_path)
        img = color.rgb2hsv(img)
        hsv = []
        for i in range(len(img.shape)):
            if len(img.shape) == 2:
                imdata = img[:, i]
            elif len(img.shape) == 3:
                imdata = img[:, :, i]
            avg = np.mean(imdata)
            hsv.append(avg)
        hsv_list.append(hsv)
    return np.mean(hsv_list, axis=0)


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


def get_features_from_post(post: InstagramPost):
    ftrs = dict(hue=0, saturation=0, value=0)
    h, s, v = get_visual_features(post.get_img_path_list())
    textual_ftrs = get_textual_features([post.caption])
    ftrs["hue"] = h
    ftrs["saturation"] = s
    ftrs["value"] = v
    ftrs = get_mean_std_from_ftrs(ftrs)
    ftrs.update(textual_ftrs)
    return ftrs