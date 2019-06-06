import torch
from sklearn.metrics import classification_report
import torch.nn as nn

class Predictor():

    def __init__(self, model, dataloader):
        self.model = model
        self.model.eval()
        self.dataloader = dataloader
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)

    def _list_from_tensor(self, tensor):
        return list(tensor.cpu().detach().numpy())

    def predict(self, threshold=0.5):
        logit_threshold = torch.tensor(threshold / (1 - threshold)).log()
        logit_threshold = logit_threshold.to(self.device)
        pred_labels = [] 
        test_labels = []
        for inputs, labels, u_name in self.dataloader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            outputs = self.model(inputs)
            preds = outputs > logit_threshold
            print(preds.size())
            print(labels.size())
            pred_labels.extend(self._list_from_tensor(preds))
            test_labels.extend(self._list_from_tensor(labels))
        report = classification_report(test_labels, pred_labels)
        print(report)