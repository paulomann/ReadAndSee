import torch
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
import torch.nn as nn
from readorsee.data.dataset import DepressionCorpus
import pandas as pd
from torch.utils.data import DataLoader
from gensim.models.fasttext import load_facebook_model
from readorsee.data import config
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from readorsee.models.training import Trainer
from readorsee.data.models import Config
import torch.optim as optim
from torch.optim import lr_scheduler

class Predictor():

    def __init__(self, model, data_type, fine_tuned):
        """ 
        model   = the model class to be instantiated, not the instantiated 
                  class itself
        """
        self.model = model
        self.embedder = self.model.__name__.lower()
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        self.data_type = data_type
        self.fine_tuned = fine_tuned
        self.configuration = Config()

    def _list_from_tensor(self, tensor):
        return list(tensor.cpu().detach().numpy())


    def predict_all(self, threshold=0.5, training_verbose=True):
        periods = [60, 212, 365]
        datasets = list(range(0,10))
        fasttext = (self.load_fasttext_model() if self.embedder == "fasttext"
                    else None)
        # res_datasets = [{str(i):[] for i in range(0,10)} for i in range(0,3)]
        results = {"60": {}, "212": {}, "365": {}}

        for days in periods:
            all_metrics = []
            for dataset in datasets:
                print("Training model for {} days and dataset {}...".format(
                    days, dataset
                ))
                model = self.instantiate_model()
                model = self.train_model(model, days, dataset,
                                         fasttext, training_verbose)
                test = DepressionCorpus(observation_period=days, 
                    subset="test", data_type=self.data_type, fasttext=fasttext, 
                    text_embedder=self.embedder, dataset=dataset)
                test_loader = DataLoader(test, batch_size=124, shuffle=True)
                metrics = self.get_metrics(model, test_loader, threshold)
                all_metrics.append(metrics)
            
            mean_metrics = np.mean(np.vstack(all_metrics), axis=0)
            results[days] = {"precision": mean_metrics[0],
                             "recall": mean_metrics[1],
                             "f1": mean_metrics[2]}
            self.print_metrics(days, mean_metrics)
        
        return results

    def get_metrics(self, model, dataloader, threshold):
        pred_data = self.predict(model, dataloader, threshold)
        Y_true, Y_pred, _, _  = zip(*pred_data)
        metrics = precision_recall_fscore_support(
                            y_true=Y_true, y_pred=Y_pred, average="binary")
        return np.array(metrics[:-1])
    
    def predict(self, model, dataloader, threshold):
        logit_threshold = torch.tensor(threshold / (1 - threshold)).log()
        logit_threshold = logit_threshold.to(self.device)
        pred_labels = []
        test_labels = []
        u_names = []
        logits = []
        for inputs, sif_weights, labels, u_name in dataloader:
            inputs = inputs.to(self.device)
            sif_weights = sif_weights.to(self.device)
            labels = labels.to(self.device)
            outputs = model(inputs, sif_weights)
            preds = outputs > logit_threshold
            pred_labels.extend(self._list_from_tensor(preds))
            test_labels.extend(self._list_from_tensor(labels))
            u_names.extend(u_name)
            logits.extend(self._list_from_tensor(outputs))

        return zip(test_labels, pred_labels, logits, u_names)

    def print_metrics(self, days, metrics):
        print("----------------------")        
        print("For Class 1 [More depressed] with {} days".format(days))
        print("\t Precision: {} \t Recall: {} \t F1: {}".format(
            metrics[0], metrics[1], metrics[2]))

    def load_fasttext_model(self):
        fasttext = load_facebook_model(
            config.PATH_TO_FASTTEXT_PT_EMBEDDINGS, encoding="utf-8")
        return fasttext
    
    def train_model(self, model, days, dataset, fasttext, verbose):

        criterion = nn.BCEWithLogitsLoss()
        parameters = filter(lambda p: p.requires_grad, model.parameters())
        general = self.configuration.general
        optimizer = getattr(self.configuration, self.data_type)["optimizer"]
        scheduler = getattr(self.configuration, self.data_type)["scheduler"]
        optimizer_ft = optim.SGD(parameters, 
                              lr=optimizer["lr"], 
                              momentum=optimizer["momentum"],
                              weight_decay=optimizer["weight_decay"],
                              nesterov=optimizer["nesterov"])
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft,
                                               step_size=scheduler["step_size"],
                                               gamma=scheduler["gamma"])
        train = DepressionCorpus(observation_period=days, subset="train",
                        data_type=self.data_type, fasttext=fasttext,
                        text_embedder=self.embedder, dataset=dataset)
        train_loader = DataLoader(train, batch_size=general["batch"], 
                                  shuffle=general["shuffle"])
        val = DepressionCorpus(observation_period=days, subset="val",
                               data_type=self.data_type, fasttext=fasttext,
                               text_embedder=self.embedder, dataset=dataset)

        val_loader = DataLoader(val, batch_size=general["batch"],
                                shuffle=general["shuffle"])

        dataloaders = {"train": train_loader, "val": val_loader}
        dataset_sizes = {"train": len(train), "val": len(val)}

        trainer = Trainer(model, dataloaders, dataset_sizes,
                          criterion, optimizer_ft,
                          exp_lr_scheduler, general["epochs"])

        trained_model = trainer.train_model(verbose)
        trained_model.eval()
        return trained_model

    def instantiate_model(self):
        model = self.model(self.fine_tuned)
        # model.eval()
        # model.to(self.device)
        return model

def plot_confusion_matrix(y_true, y_pred, classes = np.array([0, 1]),
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    metrics = precision_recall_fscore_support(
                            y_true=y_true, 
                            y_pred=y_pred,
                            average="binary")
    print()
    print("\tPrecision \t{}\n\tRecall \t{}\n\tF1-score \t{}".format(
        metrics[0], metrics[1], metrics[2]))
    #print(scores)