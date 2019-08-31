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
import pickle
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
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score, auc, roc_curve, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns


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
            if  p:
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

class ExperimentReader():
    """
    Simple class to read all saved experiments with "*.experiment" extension.
    "*.experiment" files have all metadata associated to calculate precision,
    recall and f1 scores for post and user levels.
    """
    
    def __init__(self, logits_aggregator: str, metrics: bool):
        """
        Parameters
        ----------
        logits_aggregator: metric to aggregate logits, possible values are: 
            mean, median, and vote
        metrics: True if you want to get metrics, false if you want to get Y_true, Y_guess
            logits and experiment name for each iteration
        """
        self.folder = settings.PATH_TO_EXPERIMENTS
        self.files_names = os.listdir(settings.PATH_TO_EXPERIMENTS)
        self.i = 0
        self.size = len(self.files_names)
        self.logits_aggregator = logits_aggregator
        self.metrics = metrics
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.i >= self.size:
            self.i = 0
            raise StopIteration
        else:
            file_name = self.files_names[self.i]
            with open(os.path.join(self.folder, file_name), "rb") as f:
                experiment = pickle.load(f)
            self.i += 1
            exp_name = file_name.split(".")[0]
            metadata = self.experiment_metadata(exp_name)
            result = self.get_user_metrics(experiment, exp_name)
            result.update(metadata)
            return result
        
    def experiment_metadata(self, experiment_name):
        tokens = experiment_name.split("_")
        _, days, embedder, *aggregator = tokens
        name = f"{embedder}"
        if aggregator:
            name += f"+{aggregator[0]}"
        metadata = {"name": name, "days": days}
        return metadata

    def get_user_metrics(self, experiment, exp_name):
        user_metrics = []
        Y_true_user_list = []
        Y_guess_user_list = []
        aggregated_logits_list = []
        for exp_dataset in experiment:

            if "post_params" not in exp_dataset:
                post_params = exp_dataset["user_params"]
                user_metric = precision_recall_fscore_support(
                    y_true=np.array(post_params["truth"]),
                    y_pred=np.array(post_params["guess"]),
                    average="binary"
                )
                Y_true_user_list.append(post_params["truth"])
                Y_guess_user_list.append(post_params["guess"])
                aggregated_logits_list.append(post_params["logits"])
            else:
                post_params = exp_dataset["post_params"]
                Y_true = np.array(post_params["truth"])
                Y_guess = np.array(post_params["guess"])
                logits = post_params["logits"]
                ids = post_params["id"]
                Y_true_user, Y_guess_user, aggregated_logits = self._get_user_true_guess(
                    Y_true, Y_guess, logits, ids
                )
                Y_true_user_list.append(Y_true_user)
                Y_guess_user_list.append(Y_guess_user)
                aggregated_logits_list.append(aggregated_logits)
                user_metric = precision_recall_fscore_support(
                    y_true=Y_true_user, y_pred=Y_guess_user, average="binary"
                )
            user_metrics.append(list(user_metric[0:-1]))

        user_metrics = np.concatenate(
            [np.mean(user_metrics, axis=0), np.std(user_metrics, axis=0)]
        )
        
        if self.metrics:
            result = {
                "precision_mean": user_metrics[0],
                "recall_mean": user_metrics[1],
                "f1_mean": user_metrics[2],
                "precision_std": user_metrics[3],
                "recall_std": user_metrics[4],
                "f1_std": user_metrics[5]
            }
        else:
            result = {
                "Y_true_user": Y_true_user_list,
                "Y_guess_user": Y_guess_user_list,
                "aggregated_logits": aggregated_logits_list
            }
        return result
        
    def _get_user_true_guess(self, Y_true, Y_guess, logits, ids):
        
        users = set(ids)
        Y_true_user = []
        Y_guess_user = []
        for user in users:
            
            user_indices = [i for i, x in enumerate(ids) if x == user]
            user_logits = logits[user_indices]
            user_true = Y_true[user_indices][0]
            Y_true_user.append(user_true)
            user_guess = None
            aggregated_user_logits = []
            
            if self.logits_aggregator == "median" or self.logits_aggregator == "mean":

                if self.logits_aggregator == "mean":
                    proba = np.mean(user_logits)
                elif self.logits_aggregator == "median":
                    proba = np.median(user_logits)
                
                if proba > 0.5:
                    user_guess = 1
                else:
                    user_guess = 0
                
                aggregated_user_logits.append(proba)
                Y_guess_user.append(user_guess)
                
            elif self.logits_aggregator == "vote":
                bin_count = np.bincount(Y_guess[user_indices])
                Y_guess_user.append(np.argmax(bin_count))
            else:
                raise ValueError(f"{self.logits_aggregator} is not valid.")

        assert len(Y_true_user) == len(Y_guess_user), "Incorrect shape between Y_true and Y_pred"
        return Y_true_user, Y_guess_user, aggregated_user_logits
    
def get_experiments_results_df(logits_aggregator):
    exp_reader = ExperimentReader(logits_aggregator, metrics=True)
    df = []
    for experiment_metrics in exp_reader:
        df.append(experiment_metrics)
    return pd.DataFrame(df)

def plot_roc_precision_curves_for_users(logits_aggregator):
    exp_reader = ExperimentReader(logits_aggregator, metrics=False)
    for data in exp_reader:
        Y_true = data["Y_true_user"]
        probas = data["aggregated_logits"]
        name = data["name"] + "-" + data["days"]
        datasets = list(range(0,10))
        _plot_roc_and_pr_curves(Y_true, probas, datasets, name)

def _plot_roc_and_pr_curves(
    y_tests,
    probas,
    labels,
    title,
    save_name = "GGGG",
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'C0', 'C1', 'C2']
):
    fig = plt.figure(figsize=(13,6))
    ax1 = fig.add_subplot(1,2,1)
    ax1.set_xlim([0.0,1.0])
    ax1.set_ylim([0.0,1.0])
    ax1.set_xlabel('Recall')
    ax1.set_ylabel('Precision')
    ax1.set_title('Precision-Recall Curve - ' + title)

    ax2 = fig.add_subplot(1,2,2)
    ax2.set_xlim([0.0,1.0])
    ax2.set_ylim([0.0,1.0])
    ax2.set_xlabel("False Positive Rate")
    ax2.set_ylabel("True Positive Rate")
    ax2.set_title("ROC Curve - " + title)

    for i, lc in enumerate(zip(labels,colors)):
        w, k = lc
        y_test = y_tests[i]
        pred_prob = probas[i]

        p,r,_ = precision_recall_curve(y_test,pred_prob)
        average_precision = average_precision_score(y_test, pred_prob)

        fpr_rf, tpr_rf,_ = roc_curve(y_test,pred_prob)

        roc_auc_rf = auc(fpr_rf, tpr_rf)

        ax1.plot(r,p,c=k,label= '{} (AP = {:0.2f})'.format(w, average_precision))
        ax2.plot(fpr_rf,tpr_rf,c=k,label= '{} (AUC = {:0.2f})'.format(w, roc_auc_rf))

    ax1.legend(loc='lower right', prop={'size': 13})    
    ax2.legend(loc='lower right', prop={'size': 12})
    # fig.savefig(save_name + ".pdf", dpi=300, bbox_inches="tight")

    plt.show()