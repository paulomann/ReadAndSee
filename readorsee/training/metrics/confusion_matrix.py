import numpy as np
from collections import OrderedDict
import csv
import pandas as pd
import pickle
from readorsee import settings
from readorsee.data.models import Config
import os
import glob


__all__ = ["ConfusionMatrix"]


class ConfusionMatrix:
    """Confusion matrix with metrics

    This class accumulates classification output, and tracks it in a confusion matrix.
    Metrics are available that use the confusion matrix
    """

    def __init__(self, labels):
        """Constructor with input labels

        :param labels: Either a dictionary (`k=int,v=str`) or an array of labels
        """
        if type(labels) is dict:
            self.labels = []
            for i in range(len(labels)):
                self.labels.append(labels[i])
        else:
            self.labels = labels
        nc = len(self.labels)
        self._cm = np.zeros((nc, nc), dtype=np.int)
        self._experiments = []

    def add(self, truth, guess):
        """Add a single value to the confusion matrix based off `truth` and `guess`

        :param truth: The real `y` value (or ground truth label)
        :param guess: The guess for `y` value (or assertion)
        """

        self._cm[truth, guess] += 1

    def __str__(self):
        values = []
        width = max(8, max(len(x) for x in self.labels) + 1)
        for i, label in enumerate([""] + self.labels):
            values += ["{:>{width}}".format(label, width=width + 1)]
        values += ["\n"]
        for i, label in enumerate(self.labels):
            values += ["{:>{width}}".format(label, width=width + 1)]
            for j in range(len(self.labels)):
                values += ["{:{width}d}".format(self._cm[i, j], width=width + 1)]
            values += ["\n"]
        values += ["\n"]
        return "".join(values)

    def save(self, outfile):
        ordered_fieldnames = OrderedDict(
            [("labels", None)] + [(l, None) for l in self.labels]
        )
        with open(outfile, "w") as f:
            dw = csv.DictWriter(f, delimiter=",", fieldnames=ordered_fieldnames)
            dw.writeheader()
            for index, row in enumerate(self._cm):
                row_dict = {l: row[i] for i, l in enumerate(self.labels)}
                row_dict.update({"labels": self.labels[index]})
                dw.writerow(row_dict)

    def reset(self):
        """Reset the matrix
        """
        self._cm *= 0

    def get_correct(self):
        """Get the diagonals of the confusion matrix

        :return: (``int``) Number of correct classifications
        """
        return self._cm.diagonal().sum()

    def get_total(self):
        """Get total classifications

        :return: (``int``) total classifications
        """
        return self._cm.sum()

    def get_acc(self):
        """Get the accuracy

        :return: (``float``) accuracy
        """
        return float(self.get_correct()) / self.get_total()

    def get_recall(self):
        """Get the recall

        :return: (``float``) recall
        """
        total = np.sum(self._cm, axis=1)
        total = (total == 0) + total
        return np.diag(self._cm) / total.astype(float)

    def get_support(self):
        return np.sum(self._cm, axis=1)

    def get_precision(self):
        """Get the precision
        :return: (``float``) precision
        """

        total = np.sum(self._cm, axis=0)
        total = (total == 0) + total
        return np.diag(self._cm) / total.astype(float)

    def get_mean_precision(self):
        """Get the mean precision across labels

        :return: (``float``) mean precision
        """
        return np.mean(self.get_precision())

    def get_weighted_precision(self):
        return np.sum(self.get_precision() * self.get_support()) / float(self.get_total())

    def get_mean_recall(self):
        """Get the mean recall across labels

        :return: (``float``) mean recall
        """
        return np.mean(self.get_recall())

    def get_weighted_recall(self):
        return np.sum(self.get_recall() * self.get_support()) / float(self.get_total())

    def get_weighted_f(self, beta=1):
        return np.sum(self.get_class_f(beta) * self.get_support()) / float(
            self.get_total()
        )

    def get_macro_f(self, beta=1):
        """Get the macro F_b, with adjustable beta (defaulting to F1)

        :param beta: (``float``) defaults to 1 (F1)
        :return: (``float``) macro F_b
        """
        if beta < 0:
            raise Exception("Beta must be greater than 0")
        return np.mean(self.get_class_f(beta))

    def get_class_f(self, beta=1):
        p = self.get_precision()
        r = self.get_recall()

        b = beta * beta
        d = b * p + r
        d = (d == 0) + d

        return (b + 1) * p * r / d

    def get_f(self, beta=1):
        """Get 2 class F_b, with adjustable beta (defaulting to F1)

        :param beta: (``float``) defaults to 1 (F1)
        :return: (``float``) 2-class F_b
        """
        p = self.get_precision()[1]
        r = self.get_recall()[1]
        if beta < 0:
            raise Exception("Beta must be greater than 0")
        d = beta * beta * p + r
        if d == 0:
            return 0
        return (beta * beta + 1) * p * r / d

    def get_all_metrics(self):
        """Make a map of metrics suitable for reporting, keyed by metric name

        :return: (``dict``) Map of metrics keyed by metric names
        """
        metrics = {"acc": self.get_acc()}
        # If 2 class, assume second class is positive AKA 1
        if len(self.labels) == 2:
            metrics["precision"] = self.get_precision()[1]
            metrics["recall"] = self.get_recall()[1]
            metrics["f1"] = self.get_f(1)
        else:
            metrics["mean_precision"] = self.get_mean_precision()
            metrics["mean_recall"] = self.get_mean_recall()
            metrics["macro_f1"] = self.get_macro_f(1)
            metrics["weighted_precision"] = self.get_weighted_precision()
            metrics["weighted_recall"] = self.get_weighted_recall()
            metrics["weighted_f1"] = self.get_weighted_f(1)
        return metrics

    def add_batch(self, truth, guess):
        """Add a batch of data to the confusion matrix

        :param truth: The truth tensor
        :param guess: The guess tensor
        :return:
        """
        for truth_i, guess_i in zip(truth, guess):
            self.add(truth_i, guess_i)

    def get_all_metrics_and_reset(self, truth, guess):
        self.add_batch(truth, guess)
        single_example_metrics = self.get_all_metrics()
        self.reset()
        return single_example_metrics

    def add_experiment(self, truth, guess, logits, u_id, config):
        media_type = config.general["media_type"]

        if media_type == "ftrs":
            self.add_experiment_for_ftrs(truth, guess, logits, u_id, config)
            return None

        post_params = {"truth": truth, "guess": guess, "logits": logits, "id": u_id}

        user_truth, user_guess, logits, ids = self.get_truth_and_guess_for_user(
            truth, guess, logits, u_id
        )
        user_params = {
            "truth": user_truth,
            "guess": user_guess,
            "logits": logits,
            "id": ids,
        }
        experiment = {
            "user_metrics": None,
            "post_metrics": None,
            "user_params": None,
            "post_params": None,
            "config": config.__dict__,
        }
        experiment["user_metrics"] = self.get_all_metrics_and_reset(
            user_truth, user_guess
        )
        experiment["post_metrics"] = self.get_all_metrics_and_reset(truth, guess)
        experiment["user_params"] = user_params
        experiment["post_params"] = post_params
        
        self._experiments.append(experiment)
    
    def add_experiment_for_ftrs(self, truth, guess, logits, u_id, config):
        user_params = {"truth": truth, "guess": guess, "logits": logits, "id": u_id}
        experiment = {
            "user_metrics": None,
            "user_params": None,
            "config": config.__dict__,
        }
        experiment["user_metrics"] = self.get_all_metrics_and_reset(
            truth, guess
        )
        experiment["user_params"] = user_params
        self._experiments.append(experiment)

    def get_truth_and_guess_for_user(self, post_truth, post_guess, post_logits, u_id):
        samples = np.array(list(zip(post_truth, post_guess, post_logits, u_id)))
        samples = pd.DataFrame(samples, columns=["true", "pred", "logits", "id"])
        users = np.unique(samples["id"])
        n_classes = len(np.unique(samples["true"]))

        Y_true_per_user = []
        Y_pred_per_user = []
        arr_logits = []
        for u in users:
            real_class = int(samples[samples["id"] == u]["true"].values[0])
            preds = samples[samples["id"] == u]["pred"].values.astype(int)
            logits = samples[samples["id"] == u]["logits"].values.astype(float)
            counts = np.bincount(preds)
            pred_class = len(counts) - np.argmax(counts[::-1]) - 1
            Y_true_per_user.append(real_class)
            Y_pred_per_user.append(pred_class)
            arr_logits.append(logits)

        return Y_true_per_user, Y_pred_per_user, arr_logits, users

    def get_mean_metrics_of_all_experiments(self, config):
        if not self._experiments:
            raise ValueError("There are no experiments to take the mean.")

        if config.general["media_type"] == "ftrs":
            return self.get_mean_metrics("user_metrics"), None
        
        return self.get_mean_metrics("user_metrics"), self.get_mean_metrics("post_metrics")

    def get_mean_metrics(self, name):
        precision = []
        recall = []
        f1 = []
        for exp in self._experiments:
            precision.append(exp[name]["precision"])
            recall.append(exp[name]["recall"])
            f1.append(exp[name]["f1"])
        results = {
            "precision": np.mean(precision),
            "recall": np.mean(recall),
            "f1": np.mean(f1)
        }
        return results

    def save_experiments(self, experiment_name):
        path = settings.PATH_TO_EXPERIMENTS
        if not os.path.isdir(path):
            os.mkdir(path)
        exp_file_path = os.path.join(path, experiment_name)
        with open(f"{exp_file_path}.experiment", "wb") as f:
            pickle.dump(self._experiments, f)

    def reset_experiments(self):
        self._experiments = []
    
    def delete_saved_experiments(self):
        path = settings.PATH_TO_EXPERIMENTS
        if os.path.isdir(path):
            files = glob.glob(path + os.sep + "*.experiment")
            for f in files:
                os.remove(f)
