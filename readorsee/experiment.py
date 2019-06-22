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
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from readorsee.models.training import Trainer
from readorsee.data.models import Config
import torch.optim as optim
from torch.optim import lr_scheduler
from readorsee.models.models import init_weight_xavier_uniform
from readorsee.models.inference import Predictor


class SentenceEmbeddingExperiment():
    """
    Class used to make the experiment for the sentence embeddings
    with AVG, SIF, and PMEAN aggregators.

    Fist we train the model for the parameters specified in the 
    settings.PATH_TO_CLFS_OPTIONS, next we make the predictions
    and calculate the Precision, Recall and F1 scores.

    """

    def __init__(self, model, fine_tuned):
        """ 
        model   = the model class to be instantiated, not the instantiated 
                  class itself
        fine_tuned = True for fine_tuned model
        """
        self.model = model
        self.embedder = self.model.__name__.lower()
        self.fine_tuned = fine_tuned
        self.config = Config()
        print("Configuration: ", self.config.general)

    def _list_from_tensor(self, tensor):
        return list(tensor.cpu().detach().numpy())


    def run(self, threshold=0.5, training_verbose=True, periods=[60,212,365]):
        datasets = list(range(0,10))
        fasttext = (self.load_fasttext_model() if self.embedder == "fasttext"
                    else None)
        results = {d:{} for d in periods}

        for days in periods:
            all_metrics = []
            for dataset in datasets:
                print("Training model for {} days and dataset {}...".format(
                    days, dataset
                ))
                model = self.instantiate_model()
                model = self.train_model(model,
                                         days,
                                         dataset,
                                         fasttext,
                                         training_verbose)
                test_loader = self._get_loader(days, fasttext, dataset)
                predictor = Predictor(model)
                metrics = predictor.predict(test_loader, threshold)
                all_metrics.append(metrics)
            
            mean_metrics = np.mean(np.vstack(all_metrics), axis=0)
            results[days] = {"precision": mean_metrics[0],
                             "recall": mean_metrics[1],
                             "f1": mean_metrics[2]}
            self.print_metrics(days, mean_metrics)
        
        return results

    def _get_loader(self, days, fasttext, dataset):

        test = DepressionCorpus(observation_period=days, 
                                subset="test",
                                data_type="txt",
                                fasttext=fasttext, 
                                text_embedder=self.embedder,
                                dataset=dataset)
            
        test_loader = DataLoader(test,
                                 batch_size=self.config.general["batch"],
                                 shuffle=self.config.general["shuffle"],
                                 drop_last=True)
        return test_loader

    def print_metrics(self, days, metrics):
        print("----------------------")        
        print("For Class 1 [More depressed] with {} days".format(days))
        print("\t Precision: {} \t Recall: {} \t F1: {}".format(
            metrics[0], metrics[1], metrics[2]))

    def load_fasttext_model(self):
        fasttext = load_facebook_model(
            settings.PATH_TO_FASTTEXT_PT_EMBEDDINGS, encoding="utf-8")
        return fasttext
    
    def train_model(self, model, days, dataset, fasttext, verbose):

        criterion = nn.BCEWithLogitsLoss()
        parameters = filter(lambda p: p.requires_grad, model.parameters())
        general = self.config.general
        optimizer = self.config.txt["optimizer"]
        scheduler = self.config.txt["scheduler"]
        optimizer_ft = optim.SGD(parameters, 
                              lr=optimizer["lr"], 
                              momentum=optimizer["momentum"],
                              weight_decay=optimizer["weight_decay"],
                              nesterov=optimizer["nesterov"])
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft,
                                               step_size=scheduler["step_size"],
                                               gamma=scheduler["gamma"])
        train = DepressionCorpus(observation_period=days,
                                 subset="train",
                                 data_type="txt",
                                 fasttext=fasttext,
                                 text_embedder=self.embedder,
                                 dataset=dataset)

        train_loader = DataLoader(train,
                                  batch_size=general["batch"], 
                                  shuffle=general["shuffle"],
                                  drop_last=True)
                                  
        val = DepressionCorpus(observation_period=days,
                               subset="val",
                               data_type="txt",
                               fasttext=fasttext,
                               text_embedder=self.embedder,
                               dataset=dataset)

        val_loader = DataLoader(val, 
                                batch_size=general["batch"],
                                shuffle=general["shuffle"],
                                drop_last=True)

        dataloaders = {"train": train_loader, "val": val_loader}
        dataset_sizes = {"train": len(train), "val": len(val)}

        trainer = Trainer(model,
                          dataloaders,
                          dataset_sizes,
                          criterion,
                          optimizer_ft,
                          exp_lr_scheduler,
                          general["epochs"])

        trained_model = trainer.train_model(verbose)
        return trained_model

    def instantiate_model(self):
        model = self.model(self.fine_tuned)
        model.fc.apply(init_weight_xavier_uniform)
        return model