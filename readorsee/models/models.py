from torchvision import models
import torch.nn as nn
import torch
from allennlp.modules.elmo import Elmo, batch_to_ids
from readorsee import settings
from readorsee.data.models import Config
import readorsee.features.sentence_embeddings as embed_sentence


class ImgFCBlock(nn.Module):
    def __init__(self, n_ftrs):
        super(ImgFCBlock, self).__init__()
        self.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(n_ftrs, 1))

    def forward(self, x):
        return self.fc(x).squeeze()


def freeze_resnet_layers(up_to_10, model):
    c = 0
    for child in model.children():
        c += 1
        if c < up_to_10:
            for param in child.parameters():
                param.requires_grad = False


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.config = Config.getInstance()
        media_type = self.config.general["media_type"]
        img_embedder = getattr(self.config, media_type)["img_embedder"]
        print(f"Using {img_embedder} embedder.")

        self.resnet = self.get_model(img_embedder.lower())
        freeze_resnet_layers(7, self.resnet)
        n_ftrs = self.resnet.fc.in_features
        self.resnet.fc = ImgFCBlock(n_ftrs)

    def forward(self, x):
        x = self.resnet(x)
        return x.squeeze()

    def get_model(self, embedder_name):
        print(f"===> Using {embedder_name}")
        if embedder_name == "resnext":
            return torch.hub.load("facebookresearch/WSL-Images", "resnext101_32x8d_wsl")
        else:
            return getattr(models, embedder_name)(pretrained=True)


# =======================================================================================
# =======================================================================================
# ===============================END OF VISUAL MODELS====================================
# ===============================BEGIN OF TEXT MODELS====================================
# =======================================================================================
# =======================================================================================

class TxtFCBlock(nn.Module):
    
    def __init__(self, n_ftrs, final_ftrs=1):
        super(TxtFCBlock, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(n_ftrs, n_ftrs // 2),
            nn.BatchNorm1d(n_ftrs // 2),
            nn.ReLU(),
            nn.Linear(n_ftrs // 2, final_ftrs),
        )
    def forward(self, x):
        return self.fc(x).squeeze()

class ELMo(nn.Module):

    def __init__(self):

        super(ELMo, self).__init__()
        options_path = settings.PATH_TO_ELMO_OPTIONS
        weights_path = settings.PATH_TO_ELMO_WEIGHTS

        self.embedding = Elmo(
            options_path, weights_path, 1, dropout=0.5, scalar_mix_parameters=[0, 0, 1]
        )
        self.out_ftrs = self.embedding.get_output_dim()

    def forward(self, x):
        x = self.embedding(x)
        masks = x["mask"].float()
        x = x["elmo_representations"][0]
        return {"representation": x, "masks": masks}
    

class FastText(nn.Module):

    def __init__(self):
        super(FastText, self).__init__()
        self.out_ftrs = 300

    def forward(self, x, masks):
        return {"representation": x, "masks": masks}


class BoW(nn.Module):
    
    def __init__(self):
        super(BoW, self).__init__()
    
    def forward(self, x):
        return {"representation": x}


def get_txt_embedder(txt_embedder):
    model = None
    if txt_embedder == "elmo":
        model = ELMo()
    elif txt_embedder == "FastText":
        model = FastText()
    elif txt_embedder == "FastText":
        model = BoW()
    return model

class MeanTxtClassifier(nn.Module):

    def __init__(self):
        super(MeanTxtClassifier, self).__init__()
        self.config = Config.getInstance()
        media_type = self.config.general["media_type"]
        media_config = getattr(self.config, media_type)
        txt_embedder = media_config["txt_embedder"]
        print(f"Using {txt_embedder} embedder.")

        self.model = get_txt_embedder(txt_embedder)
        self.out_ftrs = self.model.out_ftrs
        if media_config["mean"] == "pmean":
            self.out_ftrs = self.out_ftrs * len(media_config["pmean"])
        
        self.fc = TxtFCBlock(self.out_ftrs)
    
    def forward(self, x, masks=None):
        if not masks:
            res = self.model(x)
        else:
            res = self.model(x, masks)
        emb = res["representation"]
        if "masks" in res:
            masks = res["masks"]
            emb = embed_sentence.get_mean(emb, masks)
        return self.fc(emb).squeeze()
        
    def set_out_ftrs(self, out_ftrs, final_ftrs=1):
        self.fc = TxtFCBlock(out_ftrs, final_ftrs)


class MLPClassifier(nn.Module):
    def __init__(self):
        super(MLPClassifier, self).__init__()
        self.config = Config.getInstance()
        media_type = self.config.general["media_type"]
        features = getattr(self.config, media_type)["features"]
        ftrs_map_n_ftrs = {"vis_ftrs": 20, "txt_ftrs": 72, "both": 84}
        n_ftrs = ftrs_map_n_ftrs[features]
        print("FEATURES SIZE:", n_ftrs)
        self.dropout = nn.Dropout(0.5)
        self.fc = TxtFCBlock(n_ftrs)

    def forward(self, x):
        x = self.dropout(x)
        x = self.fc(x).squeeze()
        return x


class TxtImgClassifier(nn.Module):
    def __init__(self):
        super(TxtImgClassifier, self).__init__()
        self.config = Config.getInstance()
        media_type = self.config.general["media_type"]
        media_config = getattr(self.config, media_type)
        txt_embedder_name = media_config["txt_embedder"]
        use_lstm = media_config["LSTM"]
        self.common_hidden_ftrs = 64

        if not use_lstm:
            self.txt_embedder = MeanTxtClassifier()
        else:
            raise NotImplementedError("LSTM with fusion is not implemented yet.")
        self.txt_block = TxtFCBlock(self.txt_embedder.out_ftrs, self.common_hidden_ftrs)
        self.img_block = ResNet()
        self.final_fc = nn.Linear(self.common_hidden_ftrs, 1)

    def forward(self, img, txt, mask=None):
        txt = self.txt_embedder(txt, mask)
        txt = self.txt_block(txt)
        img = self.img_block(img)
        x = txt * img
        x = self.final_fc(x)
        return x.squeeze()
    
    def set_out_ftrs(self, out_ftrs):
        self.txt_embedder.set_out_ftrs(out_ftrs, self.common_hidden_ftrs)

def init_weight_xavier_uniform(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
