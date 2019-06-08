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

class Predictor():

    def __init__(self, model):
        self.model = model
        self.model.eval()
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)

    def _list_from_tensor(self, tensor):
        return list(tensor.cpu().detach().numpy())

    def _predict(self, dataloader, threshold=0.5):
        logit_threshold = torch.tensor(threshold / (1 - threshold)).log()
        logit_threshold = logit_threshold.to(self.device)
        pred_labels = [] 
        test_labels = []
        for inputs, labels, u_name in dataloader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            outputs = self.model(inputs)
            preds = outputs > logit_threshold
            pred_labels.extend(self._list_from_tensor(preds))
            test_labels.extend(self._list_from_tensor(labels))
        # report = classification_report(test_labels, pred_labels)
        # print(report)
        report = self.pandas_classification_report(test_labels, pred_labels)
        class0 = report.iloc[0].values[:3]
        class1 = report.iloc[1].values[:3]
        return class0, class1
    
    def predict(self, data_type="txt", embedder="elmo", threshold=0.5):
        periods = [60, 212, 365]
        datasets = list(range(0,10))
        fasttext = self.load_fasttext_model() if embedder == "fasttext" else None
        for days in periods:

            metrics_class0 = []
            metrics_class1 = []
            for dataset in datasets:

                test = DepressionCorpus(observation_period=days, 
                    subset="test", data_type=data_type, fasttext=fasttext, 
                    text_embedder=embedder, dataset=dataset)
                test_loader = DataLoader(test, batch_size=124, shuffle=True)
                class0, class1 = self._predict(test_loader, threshold)
                metrics_class0.append(class0)
                metrics_class1.append(class1)
            
            metrics0 = np.mean(np.vstack(metrics_class0), axis=0)
            metrics1 = np.mean(np.vstack(metrics_class1), axis=0)
            self.print_metrics(days, metrics0, metrics1)

    def print_metrics(self, days, metrics0, metrics1):

        print("----------------------")
        print("For Class 0 [Less depressed] with {} days".format(days))
        print("\t Precision: {} \t Recall: {} \t F1: {}".format(
            metrics0[0], metrics0[1], metrics0[2]))
        
        print("For Class 1 [More depressed] with {} days".format(days))
        print("\t Precision: {} \t Recall: {} \t F1: {}".format(
            metrics1[0], metrics1[1], metrics1[2]))

    def load_fasttext_model(self):
        fasttext = load_facebook_model(
            config.PATH_TO_FASTTEXT_PT_EMBEDDINGS, encoding="utf-8")
        return fasttext

    def pandas_classification_report(self, y_true, y_pred):
        metrics_summary = precision_recall_fscore_support(
                y_true=y_true, 
                y_pred=y_pred)

        avg = list(precision_recall_fscore_support(
                y_true=y_true, 
                y_pred=y_pred,
                average='weighted'))

        metrics_sum_index = ['precision', 'recall', 'f1-score', 'support']
        class_report_df = pd.DataFrame(
            list(metrics_summary),
            index=metrics_sum_index)

        support = class_report_df.loc['support']
        total = support.sum() 
        avg[-1] = total

        class_report_df['avg / total'] = avg

        return class_report_df.T