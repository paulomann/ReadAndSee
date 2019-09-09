from torchvision import models
import torch.nn as nn
import torch
from allennlp.modules.elmo import Elmo, batch_to_ids
from readorsee import settings
import readorsee.features.sentence_embeddings as embed_sentence


class ImgFCBlock(nn.Module):
    def __init__(self, n_ftrs, out_ftrs=1):
        super(ImgFCBlock, self).__init__()
        self.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(n_ftrs, out_ftrs))

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
    def __init__(self, config):
        super(ResNet, self).__init__()
        self.config = config
        media_type = self.config.general["media_type"]
        img_embedder = getattr(self.config, media_type)["img_embedder"]
        print(f"Using {img_embedder} embedder.")

        self.resnet = self.get_model(img_embedder.lower())
        freeze_resnet_layers(7, self.resnet)
        n_ftrs = self.resnet.fc.in_features
        self.out_ftrs = n_ftrs
        self.resnet.fc = ImgFCBlock(n_ftrs)

    def forward(self, x):
        x = self.resnet(x)
        return x.squeeze()

    def get_model(self, embedder_name):
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
            options_path, 
            weights_path,
            num_output_representations=1,
            dropout=0.5,
            scalar_mix_parameters=[-9e10, -9e10, 1]
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
        self.dropout = nn.Dropout(0.5)
    def forward(self, x):
        x = self.dropout(x)
        return {"representation": x}


def get_txt_embedder(txt_embedder):
    model = None
    if txt_embedder == "elmo":
        model = ELMo()
    elif txt_embedder == "fasttext":
        model = FastText()
    elif txt_embedder == "bow":
        model = BoW()
    return model


class MeanTxtClassifier(nn.Module):
    def __init__(self, config):
        super(MeanTxtClassifier, self).__init__()
        self.config = config
        media_type = self.config.general["media_type"]
        media_config = getattr(self.config, media_type)
        txt_embedder = media_config["txt_embedder"]
        print(f"Using {txt_embedder} embedder.")

        self.model = get_txt_embedder(txt_embedder)

        if hasattr(self.model, "out_ftrs"):
            self.out_ftrs = self.model.out_ftrs
            if media_config["mean"] == "pmean":
                self.out_ftrs = self.out_ftrs * len(media_config["pmean"])

            self.fc = TxtFCBlock(self.out_ftrs)

    def forward(self, x, masks=None):
        if masks is None:
            res = self.model(x)
        else:
            res = self.model(x, masks)
        emb = res["representation"]
        if "masks" in res:
            masks = res["masks"]
            emb = embed_sentence.get_mean(emb, masks, self.config)
        return self.fc(emb).squeeze()

    def set_out_ftrs(self, out_ftrs, final_ftrs=1):
        self.fc = TxtFCBlock(out_ftrs, final_ftrs)


class LSTMTxtClassifier(nn.Module):
    def __init__(self, config):
        super(LSTMTxtClassifier, self).__init__()
        self.config = config
        media_type = self.config.general["media_type"]
        media_config = getattr(self.config, media_type)
        txt_embedder = media_config["txt_embedder"]
        print(f"Using {txt_embedder} embedder.")
        if txt_embedder not in ["elmo", "fasttext"]:
            raise ValueError("LSTMClassifier only supports elmo and fasttext.")
        self.hidden_units = 64
        self.embedder = get_txt_embedder(txt_embedder)
        self.embed_dims = self.embedder.out_ftrs
        self.num_layers = media_config["LSTM"]["num_layers"]
        self.bidirectional = media_config["LSTM"]["bidirectional"]
        self.num_directions = 2 if self.bidirectional else 1
        self.out_ftrs = self.hidden_units * self.num_directions
        self.dropout_value = media_config["LSTM"]["dropout"]
        print(f"===LSTM params {media_config['LSTM']}")
        self.lstm = nn.LSTM(
            self.embed_dims,
            self.hidden_units,
            bidirectional=self.bidirectional,
            num_layers=self.num_layers,
            dropout=self.dropout_value,
        )
        self.fc = nn.Linear(self.out_ftrs, 1)
        nn.init.orthogonal_(self.lstm.weight_hh_l0)
        nn.init.orthogonal_(self.lstm.weight_ih_l0)

    def forward(self, x, masks=None):
        if masks is None:
            res = self.embedder(x)
        else:
            res = self.embedder(x, masks)
        emb = res["representation"]
        if "masks" in res:
            masks = res["masks"]
        lengths = masks.sum(dim=1)
        bsz = len(lengths)
        lengths, perm_idx = lengths.sort(0, descending=True)
        orig_idx = sorted(range(len(perm_idx)), key=perm_idx.__getitem__)
        emb = emb[perm_idx]
        emb = emb.permute(1, 0, 2)
        packed = torch.nn.utils.rnn.pack_padded_sequence(emb, lengths.tolist())
        _, (hidden, _) = self.lstm(packed)

        if self.bidirectional:

            def combine_bidir(outs):
                return torch.cat(
                    [
                        torch.cat([outs[2 * i], outs[2 * i + 1]], dim=0).view(
                            1, bsz, self.out_ftrs
                        )
                        for i in range(self.num_layers)
                    ],
                    dim=0,
                )
            hidden = combine_bidir(hidden)
        # Always take the output from the last LSTM layer
        hidden = hidden[-1]
        linear = self.fc(hidden)
        linear = linear[orig_idx]
        return linear.squeeze()


class MLPClassifier(nn.Module):
    def __init__(self, config):
        super(MLPClassifier, self).__init__()
        self.config = config
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


class MultimodalClassifier(nn.Module):
    def __init__(self, config):
        super(MultimodalClassifier, self).__init__()
        self.config = config
        media_type = self.config.general["media_type"]
        media_config = getattr(self.config, media_type)
        use_lstm = media_config["LSTM"]
        self.common_hidden_units = 64

        if not use_lstm:
            self.txt_embedder = MeanTxtClassifier(self.config)
        else:
            raise NotImplementedError("LSTM classification is not yet implemented.")

        if hasattr(self.txt_embedder, "out_ftrs"):
            self.txt_embedder.fc = TxtFCBlock(
                self.txt_embedder.out_ftrs, self.common_hidden_units
            )
        self.img_embedder = ResNet(self.config)
        self.img_embedder.resnet.fc = ImgFCBlock(
            self.img_embedder.out_ftrs, self.common_hidden_units
        )
        self.final_fc = nn.Linear(self.common_hidden_units, 1)

    def forward(self, img, txt, mask=None):
        txt = self.txt_embedder(txt, mask)
        img = self.img_embedder(img)
        x = txt * img
        x = self.final_fc(x)
        return x.squeeze()

    def set_out_ftrs(self, out_ftrs):
        self.txt_embedder.set_out_ftrs(out_ftrs, self.common_hidden_units)