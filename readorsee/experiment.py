import torch
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
import torch.nn as nn
from readorsee.data.dataset import DepressionCorpus
import pandas as pd
from torch.utils.data import DataLoader
from gensim.models.fasttext import load_facebook_model
from readorsee import settings
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from readorsee.training import train_model
from readorsee.data.models import Config
import torch.optim as optim
from torch.optim import lr_scheduler
from readorsee.models.models import init_weight_xavier_uniform
from readorsee.models.inference import Predictor
from readorsee.training.metrics import ConfusionMatrix
import readorsee.models.models as models


class DetectDepressionExperiment():
    """
    Class used to make the experiment for the sentence embeddings
    with AVG, SIF, and PMEAN aggregators.

    Fist we train the model for the parameters specified in the 
    settings.PATH_TO_CLFS_OPTIONS, next we make the predictions
    and calculate the Precision, Recall and F1 scores, saving
    the experiments data in a *.experiment file, which is an 
    object stored with pickle.

    """

    def __init__(self):
        self.config = Config()
        media_type = self.config.general["media_type"]
        self.media_config = getattr(self.config, media_type)
        self.media_type = media_type
        model_name = self.media_config["embedder"]
        self.model = getattr(models, model_name)
        self.embedder = self.model.__name__.lower()
        print("======================")
        print(f"Using {self.model.__name__} model")
        print(f"General Configuration: {self.config.general}")
        print(f"Media Configuration: {self.media_config}")
        print("======================")

    def _list_from_tensor(self, tensor):
        return list(tensor.cpu().detach().numpy())

    def run(self, 
            threshold=0.5,
            training_verbose=True,
            periods=[60, 212, 365]
        ):

        datasets = list(range(0,10))
        fasttext = (self.load_fasttext_model() if self.embedder == "fasttext"
                    else None)
        results = {self.media_type:{d:{} for d in periods}}
        cm = ConfusionMatrix([0,1])

        print(f"===Using {self.media_type} media")
        for days in periods:
            for dataset in datasets:
                print(f"Training model for {days} days and dataset {dataset}...")
                model = self.instantiate_model()
                model = train_model(model,
                                    days,
                                    dataset,
                                    fasttext,
                                    self.config,
                                    training_verbose)
                test_loader = self._get_loader(days, fasttext, dataset)
                predictor = Predictor(model)
                cm = predictor.predict(test_loader, cm, threshold)
                self.free_model_memory(model)
            experiment_name = self.get_experiment_name(self.media_type, days)
            print(experiment_name)
            user_results, post_results = cm.get_mean_metrics_of_all_experiments()
            cm.save_experiments(experiment_name)
            cm.reset_experiments()
            results[self.media_type][days] = {"user": user_results, "post": post_results}
            print(f"===>For Class 1 [More depressed] with {days} days")
            print(f"{results[self.media_type][days]}")

        self.print_metrics(results, self.media_type)

        return results

    def _get_loader(self, days, fasttext, dataset):

        test = DepressionCorpus(observation_period=days, 
                                subset="test",
                                fasttext=fasttext,
                                dataset=dataset)
            
        test_loader = DataLoader(test,
                                 batch_size=self.config.general["batch_size"],
                                 shuffle=self.config.general["shuffle"],
                                 pin_memory=True)
        return test_loader
    
    def free_model_memory(self, model):
        del model
        torch.cuda.empty_cache()
        if self.embedder == "bow": os.remove(settings.PATH_TO_SERIALIZED_TFIDF)
    
    def get_experiment_name(self, media_type, days):
        media_config = getattr(self.config, media_type)
        embedder = ""
        if isinstance(media_config["embedder"], str):
            embedder = media_config["embedder"]
        elif isinstance(media_config["embedder"], list):
            embedder = "+".join(media_config["embedder"])
        embedder = embedder.lower()
        aggregator = media_config.get("mean", "")
        exp_name = f"{media_type}_{days}_{embedder}"

        if aggregator and embedder != "bow":
            exp_name = exp_name + f"_{aggregator}"
        if media_type == "ftrs":
            features = media_config["features"].replace("_", "-")
            exp_name = exp_name + f"_{features}"

        return exp_name

    def print_metrics(self, results, media_type):
        print(f"=====================>For {media_type}")
        metrics = results[media_type]
        for k, v in metrics.items():
            print(f"===>For Class 1 [More depressed] with {k} days")
            u = v["user"]
            p = v["post"]
            print(f">User:\n\t Precision: {u['precision']} \t Recall: {u['recall']}" \
                  f"\t F1: {u['f1']}")
            print(f">Post:\n\t Precision: {p['precision']} \t Recall: {p['recall']}" \
                  f"\t F1: {p['f1']}")

    def load_fasttext_model(self):
        fasttext = load_facebook_model(
            settings.PATH_TO_FASTTEXT_PT_EMBEDDINGS, encoding="utf-8")
        return fasttext

    def instantiate_model(self):
        model = self.model()
        if hasattr(model, "fc"):
            model.fc.apply(init_weight_xavier_uniform)
        return model