from torchvision import models
import torch.nn as nn
import torch
from allennlp.modules.elmo import Elmo, batch_to_ids
from readorsee import settings
from readorsee.data.models import Config
from readorsee.features.sentence_embeddings import SIF, PMEAN

class ResNet50(nn.Module):

    def __init__(self):
        super(ResNet50, self).__init__()
        self.resnet = models.resnet50(pretrained=True)

        # Freezing all layers but layer3, layer4 and avgpool
        c = 0
        for child in self.resnet.children():
            c += 1
            if c < 7:
                for param in child.parameters():
                    param.requires_grad = False

        n_ftrs = self.resnet.fc.in_features

        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(n_ftrs, 1)
        )

    def forward(self, x):
        x = self.resnet(x)
        return x.squeeze()

class ResNet34(nn.Module):

    def __init__(self):
        super(ResNet34, self).__init__()
        self.resnet = models.resnet34(pretrained=True)

        # Freezing all layers but layer3, layer4 and avgpool
        c = 0
        for child in self.resnet.children():
            c += 1
            if c < 7:
                for param in child.parameters():
                    param.requires_grad = False

        n_ftrs = self.resnet.fc.in_features

        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(n_ftrs, 1)
        )

    def forward(self, x):
        x = self.resnet(x)
        return x.squeeze()

class ResNet18(nn.Module):

    def __init__(self):
        super(ResNet18, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        c = 0
        for child in self.resnet.children():
            c += 1
            if c < 7:
                for param in child.parameters():
                    param.requires_grad = False

        n_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(n_ftrs, 1)
        )

    def forward(self, x):
        x = self.resnet(x)
        return x.squeeze()

class ResNext(nn.Module):
    """
    RexNext pre-trained with 940 million Instagram public images to predict
    hashtags of these social media images.
    """
    def __init__(self):
        super(ResNext, self).__init__()
        self.resnext = torch.hub.load(
            'facebookresearch/WSL-Images', 'resnext101_32x8d_wsl'
        )
        c = 0
        for child in self.resnext.children():
            c += 1
            if c < 7:
                for param in child.parameters():
                    param.requires_grad = False
        
        n_ftrs = self.resnext.fc.in_features

        self.resnext.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(n_ftrs, 1)
        )
    
    def forward(self, x):
        x = self.resnext(x)
        return x.squeeze()

class ELMo(nn.Module):

    def __init__(self, fine_tuned=False):
        """
        fine_tuned = if False uses the ELMo trained on the wikipedia PT-BR dump.
                     Otherwise uses the ELMo trained on the wikipedia tuned 
                     with our 31 million tweets dataset.

        """
        super(ELMo, self).__init__()
        self.fine_tuned = fine_tuned

        self.configuration = Config()

        options_path = (settings.PATH_TO_FT_ELMO_OPTIONS if fine_tuned else
                        settings.PATH_TO_ELMO_OPTIONS)
        weights_path = (settings.PATH_TO_FT_ELMO_WEIGHTS if fine_tuned else
                        settings.PATH_TO_ELMO_WEIGHTS)

        self.embedding = Elmo(options_path, weights_path, 1, dropout=0.5,
                              scalar_mix_parameters=[0, 0, 1])

        n_ftrs = self.embedding.get_output_dim()

        if self.configuration.txt["mean"] == "pmean":
            n_ftrs = n_ftrs * len(self.configuration.txt["pmean"])

        self.fc = nn.Sequential(
            nn.Linear(n_ftrs, n_ftrs//2),
            nn.BatchNorm1d(n_ftrs//2),
            nn.ReLU(),
            nn.Linear(n_ftrs//2, 1)
        )
        # self._init_weight()

    def forward(self, x, sif_weights=None):
        x = self.embedding(x)
        masks = x["mask"].float()
        x = x["elmo_representations"][0]
        # ----------------------------------------------------
        x = self._get_mean(x, masks, sif_weights)
        # ----------------------------------------------------
        x = self.fc(x)
        x = x.squeeze()
        return x

    def _get_mean(self, x, masks, sif_weights):
        
        if self.configuration.txt["mean"] == "sif":
            sif = SIF()
            sif_embeddings = sif.SIF_embedding(x, masks, sif_weights)
            return sif_embeddings

        elif self.configuration.txt["mean"] == "pmean":
            pmean = PMEAN()
            means = self.configuration.txt["pmean"]
            pmean_embedding = pmean.PMEAN_embedding(x, masks, means)
            return pmean_embedding

        elif self.configuration.txt["mean"] == "avg":
            x = x.sum(dim=1)
            masks = masks.sum(dim=1).view(-1, 1).float()
            x = torch.div(x, masks)
            x[torch.isnan(x)] = 0
            x[torch.isinf(x)] = 1
            return x
        else:
            raise NotImplementedError


class FastText(nn.Module):
    def __init__(self, fine_tuned=False):
        """ 
        fine_tuned   = use the fine_tuned model <<Not implemented yet>>
        """
        super(FastText, self).__init__()

        self.fine_tuned = fine_tuned
        self.config = Config()
        n_ftrs = 300

        if self.config.txt["mean"] == "pmean":
            n_ftrs = n_ftrs * len(self.config.txt["pmean"])

        self.fc = nn.Sequential(
            nn.Linear(n_ftrs, n_ftrs//2),
            nn.BatchNorm1d(n_ftrs//2),
            nn.ReLU(),
            nn.Linear(n_ftrs//2, 1)
        )

    def forward(self, x):
        x = self.fc(x).squeeze()
        return x

    def init_weight(self, dataset, days):
        pass

def init_weight_xavier_uniform(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)


class BoW(nn.Module):

    def __init__(self):
        super(BoW, self).__init__()

    def forward(self, x):
        x = self.fc(x).squeeze()
        return x
    
    def create_layers(self, n_ftrs):
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(n_ftrs, n_ftrs//2),
            nn.BatchNorm1d(n_ftrs//2),
            nn.ReLU(),
            nn.Linear(n_ftrs//2, 1)
        )


class MLPClassifier(nn.Module):
    def __init__(self):
        super(MLPClassifier, self).__init__()
        self.config = Config()
        data_type = self.config.general["media_type"]
        features = getattr(self.config, data_type)["features"]
        ftrs_map_n_ftrs = {"vis_ftrs": 20, "txt_ftrs": 72, "both": 84}
        n_ftrs = ftrs_map_n_ftrs[features]
        print("FEATURES SIZE:", n_ftrs)
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(n_ftrs, n_ftrs//2),
            nn.BatchNorm1d(n_ftrs//2),
            nn.ReLU(),
            nn.Linear(n_ftrs//2, 1)
        )        

    def forward(self, x):
        x = self.fc(x).squeeze()
        return x

class FusionModel(nn.Module):
    def __init__(self):
        super(FusionModel, self).__init__()
        self.configuration = Config()

        options_path = settings.PATH_TO_ELMO_OPTIONS
        weights_path = settings.PATH_TO_ELMO_WEIGHTS

        self.embedding = Elmo(options_path, weights_path, 1, dropout=0.5,
                              scalar_mix_parameters=[0, 0, 1])

        self.resnet = models.resnet18(pretrained=True)
        img_ftrs = self.resnet.fc.in_features
        c = 0
        for child in self.resnet.children():
            c += 1
            if c < 7:
                for param in child.parameters():
                    param.requires_grad = False

        self.resnet.fc = nn.Sequential(
            nn.Linear(img_ftrs, img_ftrs//2),
            nn.BatchNorm1d(img_ftrs//2),
            nn.ReLU(),
            nn.Linear(img_ftrs//2, 64)
        )
        txt_ftrs = self.embedding.get_output_dim()

        if self.configuration.both["mean"] == "pmean":
            txt_ftrs = txt_ftrs * len(self.configuration.both["pmean"])

        self.txt_embedding = nn.Sequential(
            nn.Linear(txt_ftrs, txt_ftrs//2),
            nn.BatchNorm1d(txt_ftrs//2),
            nn.ReLU(),
            nn.Linear(txt_ftrs//2, 64)
        )

        self.fc = nn.Linear(64, 1)

    def forward(self, txt, img):
        txt = self.embedding(txt)
        masks = txt["mask"].float()
        txt = txt["elmo_representations"][0]
        txt = self._get_mean(txt, masks)
        txt = self.txt_embedding(txt)
        img = self.resnet(img)
        x = txt * img
        self.fc(x)
        return x.squeeze()

    def _get_mean(self, x, masks):
        
        if self.configuration.both["mean"] == "pmean":
            pmean = PMEAN()
            means = self.configuration.txt["pmean"]
            pmean_embedding = pmean.PMEAN_embedding(x, masks, means)
            return pmean_embedding

        elif self.configuration.both["mean"] == "avg":
            x = x.sum(dim=1)
            masks = masks.sum(dim=1).view(-1, 1).float()
            x = torch.div(x, masks)
            x[torch.isnan(x)] = 0
            x[torch.isinf(x)] = 1
            return x
        else:
            raise NotImplementedError