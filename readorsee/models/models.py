from torchvision import models
import torch.nn as nn
import torch
from allennlp.modules.elmo import Elmo, batch_to_ids
from readorsee.data import config


class ResNet(nn.Module):

    def __init__(self, resnet_size=50, n_classes=2):
        super(ResNet, self).__init__()

        self.resnet = getattr(models, "resnet" + str(resnet_size))
        self.resnet = self.resnet(pretrained=True)

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
            nn.Linear(n_ftrs, n_classes)
        )

    def forward(self, x):
        x = self.resnet(x)
        return x


class ELMo(nn.Module):

    def __init__(self, n_classes=2):
        super(ELMo, self).__init__()

        self.embedding = Elmo(config.PATH_TO_ELMO_OPTIONS,
                              config.PATH_TO_ELMO_WEIGHTS, 1, dropout=0.2)
        n_ftrs = self.embedding.get_output_dim()
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(n_ftrs, n_classes)
        )

    def forward(self, x):
        x = batch_to_ids(x)
        x = self.embedding(x)["elmo_representations"][0]
        # This is where we get the mean of the word embeddings
        x = torch.mean(x, dim=1)
        x = self.fc(x)
        return x

# Hot to
# gg = elmo([["Eu", "quero", "!"], ["Aprender", "deep", "learning"]])
