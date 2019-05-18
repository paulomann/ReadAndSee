from torchvision import models
import torch.nn as nn
import torch
from allennlp.modules.elmo import Elmo, batch_to_ids
from readorsee.data import config
from gensim.models.fasttext import load_facebook_model


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
        print("\tIn Model: input size", x.size())
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
        x = [sentence.split(" ") for sentence in x]
        print("\tIn Model: input size", len(x))
        x = batch_to_ids(x)
        x = self.embedding(x)["elmo_representations"][0]
        # This is where we get the mean of the word embeddings
        x = torch.mean(x, dim=1)
        x = self.fc(x)
        return x

class FastText(nn.Module):
    
    def __init__(self, n_classes=2):
        fasttext = self.load_fasttext_model()

        self.embedding = fasttext.wv
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(fasttext.vector_size, n_classes)
        )

    def forward(self, x):
        x = [sentence.split(" ") for sentence in x]
        for sentence in x:
            embedding = torch.mean([self.embedding[tok] for tok in sentence])
            # TODO: append the embeddings mean altogether

    def load_fasttext_model(self):
        fasttext = load_facebook_model(
            config.PATH_TO_FASTTEXT_PT_EMBEDDINGS, encoding="utf-8")
        return fasttext
        

# elmo = ELMo()
# gg = elmo(["eu gosto", "de deep learning"])

FastText()




    # def load_vectors(self):
    #     fname = config.PATH_TO_FASTTEXT_PT_EMBEDDINGS
    #     fin = io.open(fname, "r", encoding="utf-8", 
    #                   newline="\n", errors="ignore")
    #     vocab_size, embed_dimension = map(int, fin.readline().split())
    #     data = {}
    #     embeddings = torch.zeros((vocab_size, embed_dimension))
    #     vocab = Counter()
    #     for i, line in enumerate(fin):
    #         tokens = line.rstrip().split(' ')
    #         embedding = torch.tensor(list(map(float, tokens[1:])))
    #         word = tokens[0]
    #         data[word] = embedding
    #         embeddings[i] = embedding
    #         vocab[word] = i
    #     return data
