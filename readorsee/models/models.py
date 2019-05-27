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

    def __init__(self, fine_tune=False, n_classes=2):
        super(ELMo, self).__init__()

        self.embedding = Elmo(config.PATH_TO_ELMO_OPTIONS,
                              config.PATH_TO_ELMO_WEIGHTS, 1, dropout=0.2,
                              requires_grad=fine_tune)
        n_ftrs = self.embedding.get_output_dim()
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(n_ftrs, n_classes)
        )

    def forward(self, x):
        x = self.embedding(x)
        mask = x["mask"]
        x = x["elmo_representations"][0]
        # This is where we get the mean of the word embeddings
        x = self._get_mean(x, mask)
        # ----------------------------------------------------
        x = self.fc(x)
        return x
        
    def _get_mean(self, x, mask):
        x = x.sum(dim=1)
        with torch.no_grad():
            mask = mask.sum(dim=1).float()
            mask = torch.repeat_interleave(mask, 
                        x.size(-1)).view(-1, x.size(-1))


class FastText(nn.Module):
    def __init__(self, n_classes=2):
        super(FastText, self).__init__()
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(300, n_classes)
        )

    def forward(self, x):
        x = self.fc(x)
        return x

def init_weight_xavier_uniform(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)