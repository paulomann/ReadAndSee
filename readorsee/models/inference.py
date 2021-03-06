import torch
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
import torch.nn as nn
from readorsee.data.dataset import DepressionCorpus
from readorsee.training.metrics import ConfusionMatrix
import pandas as pd
from torch.utils.data import DataLoader
from gensim.models.fasttext import load_facebook_model
from readorsee import settings
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
import torch.optim as optim
from torch.optim import lr_scheduler
from scipy.special import expit

class Predictor():
    """
    Predictor for binary classification problems. 
    """

    def __init__(self, model, config):
        """ 
        model   = the model class to be instantiated, not the instantiated 
                  class itself
        """
        self.model = model
        self.configuration = config
        general_config = self.configuration.general
        gpus = general_config["gpus"]
        self.device = torch.device(
            f"cuda:{gpus[0]}" if torch.cuda.is_available() else "cpu")
        if not next(model.parameters()).is_cuda:
            self.model = self.model.to(self.device)

    def _list_from_tensor(self, tensor):
        if tensor.numel() == 1:
            return [tensor.item()]
        return list(tensor.cpu().detach().numpy())
    
    def predict(self, dataloader, cm, threshold=0.5):
        self.model.eval()
        logit_threshold = torch.tensor(threshold / (1 - threshold)).log()
        logit_threshold = logit_threshold.to(self.device)
        pred_labels = []
        test_labels = []
        u_names = []
        logits = []
        for *inputs, labels, u_name in dataloader:
            inputs = [i.to(self.device) for i in inputs]
            labels = labels.to(self.device)
            outputs = self.model(*inputs)
            preds = outputs > logit_threshold
            pred_labels.extend(self._list_from_tensor(preds))
            test_labels.extend(self._list_from_tensor(labels))
            u_names.extend(u_name)
            logits.extend(self._list_from_tensor(outputs))
        #Take probabilities and not logit
        logits = expit(logits)
        cm.add_experiment(test_labels, pred_labels, logits, u_names, self.configuration)
        return cm


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
