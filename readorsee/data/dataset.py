from readorsee.data.facade import StratifyFacade
from readorsee.data import config
from readorsee.data.models import Config
import torch
from torchvision import transforms
import torchvision
import pandas as pd
import numpy as np
from PIL import Image
import os
from readorsee.data.preprocessing import Tokenizer
from gensim.models.fasttext import load_facebook_model
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from allennlp.modules.elmo import batch_to_ids
from collections import defaultdict


class DepressionCorpus(torch.utils.data.Dataset):

    def __init__(self, observation_period, dataset,
                 subset, data_type, fasttext=None,
                 text_embedder="", transform=None):
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
                raise ValueError("{} is not a valid embedder"
                                .format(text_embedder))

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

        self.configuration = Config()

        self.word_freqs = self.get_word_frequencies()
        self.sif_weights = self.get_word_SIF_weights()

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

        if self.text_embedder == "elmo":
            caption = self._elmo[idx]
        elif self.text_embedder == "fasttext":
            caption = self._fasttext[idx]

        if self._data_type == "txt":
            data = (caption,)
        elif self._data_type == "both":
            data = (img, caption)
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
                images_paths = [os.path.join(config.PATH_TO_INSTAGRAM_DATA, p)
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
        # text = text[:600].strip()
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

        def get_mean(x, mask, sentences):
            # if self.configuration.general["mean"] == "sif":
            #     ftrs = mask.size(1)
            #     #sif_weights = torch.ones_like(x.size())
            #     sif_weights = []
            #     for sent in sentences:
            #         sif_sent = [self.sif_weights[token] for token in sent]
            #         sif_sent = torch.tensor(sif_sent)
            #         sent_size = sif_sent.size(0)
            #         sif_sent = torch.cat(
            #             [sif_sent, torch.zeros(ftrs - sent_size)])
            #         sif_weights.append(sif_sent)
                
            #     sif_weights = torch.stack(sif_weights)
            #     assert sif_weights.size(0) == x.size(0)
            #     assert sif_weights.size(1) == x.size(1)
                

            x = x.sum(dim=1)
            mask = mask.sum(dim=1).float()
            mask = torch.repeat_interleave(mask, 
                        x.size(-1)).view(-1, x.size(-1))
            x = torch.div(x, mask)
            return x

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

        embeddings = get_mean(embeddings, masks, texts)

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
            questionnaire_answers = []
            for profile in participants:
                answer_dict, keys = profile.get_answer_dict()
                questionnaire_answers.append(answer_dict)

            df = pd.DataFrame(questionnaire_answers, columns=cols_order + keys)
            return df

        return get_answers_df(subset)

    def _load_instagram_questionnaire_answers(self):
        answers_path = os.path.join(config.PATH_TO_INTERIM_DATA,
                                    "instagram.csv")
        return pd.read_csv(answers_path, encoding="utf-8")
    
    def get_word_frequencies(self):
        _, sentences, _, _ = zip(*self._data)
        
        word_freqs = defaultdict(int)
        vocab_size = 0
        for sent in sentences:
            for token in sent:
                word_freqs[token] += 1
        
        for k, v in word_freqs.items():
            vocab_size += v
        for k, v in word_freqs.items():
            word_freqs[k] = v/vocab_size

        return word_freqs

    def get_word_SIF_weights(self, a=1e-3):
        _, sentences, _, _ = zip(*self._data)
        word2weight = {}
        for sent in sentences:
            for token in sent:
                word2weight[token] = a / (a + self.word_freqs[token])
        return word2weight

# ds = DepressionCorpus(60, 0, "train", "txt", None, "elmo")

# How to use
# dc = DepressionCorpus(observation_period=60, dataset=0, subset="train",
#                       data_type="img")
# dl = DataLoader(dc, batch_size=4, shuffle=True)
# dataiter = iter(dl)
# text, labels = dataiter.next()

def imshow(inp, *args):
    """ Inp is a batch of images' tensors """
    inp = torchvision.utils.make_grid(inp) 
    for i in args:
        print(i)
    inp = inp.numpy().transpose((1, 2, 0))  # H x W x C
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    plt.show()


# TODO: Reimplement those methods using the DepressionCorpus class to get
# statistics about the data

    # def get_most_similar_datasets(self, tr_frac=0.6, val_frac=0.2,
    #                               test_frac=0.2):
    #     """ Return a dict containing the index of the best dataset for every
    #     obs. period considered.

    #     Return:
    #         A dict like {"data_<obs.period>": index}
    #     """
    #     bests = {}
    #     self._raw = self._raw if self._raw else self.load_raw_data()

    #     for key, value in self._raw.items():
    #         days = int(key.split("_")[1])
    #         dataset = self._raw[key][0]
    #         bests[key] = None
    #         original_ds = np.concatenate([dataset[0], dataset[1], dataset[2]])
    #         original_bdi_0_frac, original_bdi_1_frac, total_qty = (
    #                 self._calculate_bdi_qty(original_ds, days))

    #         minimum = [100000000, -1]
    #         for i, dataset in enumerate(self._raw[key]):
    #             train_bdi_0_frac, _, tr_qty = (
    #                 self._calculate_bdi_qty(dataset[0], days))
    #             val_bdi_0_frac, _, val_qty = (
    #                 self._calculate_bdi_qty(dataset[1], days))
    #             test_bdi_0_frac, _, test_qty = (
    #                 self._calculate_bdi_qty(dataset[2], days))

    #             bdis_fracs = [np.abs(train_bdi_0_frac - original_bdi_0_frac),
    #                           np.abs(test_bdi_0_frac - original_bdi_0_frac),
    #                           np.abs(val_bdi_0_frac - original_bdi_0_frac)]
    #             delta = sum(bdis_fracs)
    #             qty_fracs = [np.abs(tr_frac - tr_qty/total_qty),
    #                          np.abs(val_frac - val_qty/total_qty),
    #                          np.abs(test_frac - test_qty/total_qty)]
    #             delta += sum(qty_fracs)

    #             if delta < minimum[0]:
    #                 minimum[0] = delta
    #                 minimum[1] = i
    #         bests[key] = minimum[1]
    #     return bests

    # def _calculate_bdi_qty(self, subset, days):

    #     def get_total_number_of_images(posts):
    #         total = 0
    #         for p in posts:
    #             total += len(p.get_img_path_list())
    #         return total

    #     bdi_fraction = {0: 0, 1: 0}
    #     for participant in subset:
    #         posts = participant.get_posts_from_qtnre_answer_date(days)
    #         # qty = len(posts)
    #         qty = get_total_number_of_images(posts)
    #         bdi = participant.questionnaire.get_binary_bdi()
    #         bdi_fraction[bdi] += qty
    #     total = (bdi_fraction[0] + bdi_fraction[1])
    #     return bdi_fraction[0]/total, bdi_fraction[1]/total, total

    # def print_image_dimensions_means(self):
    #     self.load_raw_data()
    #     for key, value in self._raw.items():
    #         print("-------------------------")
    #         print("DATASET: {}".format(key))
    #         dataset = self._raw[key][0]
    #         train, val, test = dataset[0], dataset[1], dataset[2]
    #         data = np.concatenate([train, val, test])
    #         width = height = []
    #         for user in data:
    #             for post in user.posts:
    #                 paths = post.get_img_path_list()
    #                 for p in paths:
    #                     img_path = os.path.join(
    #                         config.PATH_TO_INSTAGRAM_DATA, p)
    #                     im = Image.open(img_path)
    #                     w, h = im.size
    #                     width.append(w)
    #                     height.append(h)

    #         print("WIDTH MEAN: {}\nHEIGHT MEAN: {}".format(
    #             sum(width)/len(width), sum(height)/len(height)))