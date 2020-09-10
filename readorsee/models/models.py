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
            scalar_mix_parameters=[-9e10, 1, -9e10]
        )
        # scalar_mix_parameters=[-9e10, -9e10, 1]
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
        self.hidden_units = 124
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
        use_lstm = media_config["use_lstm"]
        self.common_hidden_units = 64

        if not use_lstm:
            self.txt_embedder = MeanTxtClassifier(self.config)
        else:
            self.txt_embedder = LSTMTxtClassifier(self.config)

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

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

class MultimodalConcatClassifier(nn.Module):
    def __init__(self, config):
        super(MultimodalConcatClassifier, self).__init__()
        self.config = config
        media_type = self.config.general["media_type"]
        media_config = getattr(self.config, media_type)
        use_lstm = media_config["use_lstm"]
        self.img_embedder = ResNet(self.config)
        self.img_embedder.resnet.fc = Identity()
        if not use_lstm:
            self.txt_embedder = MeanTxtClassifier(self.config)
        else:
            self.txt_embedder = LSTMTxtClassifier(self.config)
        self.txt_embedder.fc = Identity()
        self.out_ftrs = self.img_embedder.out_ftrs + self.txt_embedder.out_ftrs
        print("MULTIMODAL OUT FEATURES:", self.out_ftrs)
        self.final_fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(self.out_ftrs, 1))

    def forward(self, img, txt, mask=None):
        with torch.no_grad():
            txt = self.txt_embedder(txt, mask)
            img = self.img_embedder(img)
            x = torch.cat([txt,img], dim=1)
        x = self.final_fc(x)
        return x.squeeze()        


# =======================================================================================
# =======================================================================================
# =======================================================================================
# ====================================== BERT ===========================================
# =======================================================================================
# =======================================================================================



import math
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss
import numpy as np
from transformers.modeling_bert import BertPooler, BertModel, BertSelfAttention, BertForSequenceClassification, BertLayerNorm

class BertPoolerBase(BertPooler):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(0.1)  # this we don't have in default BertPooler
        self.distribution = "normal"
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.pooler_activation = nn.Tanh()
        # if config.pooler_activation == 'tanh':
        #     self.pooler_activation = nn.Tanh()
        # elif config.pooler_activation == 'relu':
        #     self.pooler_activation = nn.ReLU()
        # elif config.pooler_activation == 'gelu':
        #     self.pooler_activation = F.gelu

    def reset_parameters(self):
        print(f'Re-initializing pooler weights from {self.distribution} distribution')

        bound = 0.03125
        if self.distribution == 'uniform':
            self.dense.weight.data.uniform_(-bound, bound)
            self.dense.bias.data.uniform_(-bound, bound)
            # self.dense.bias.data.zero_()
        elif self.distribution == 'normal':
            # BERT initializes linear layers with: module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # where self.config.initializer_range = 0.02
            self.dense.weight.data.normal_(mean=0.0, std=0.02)
            self.dense.bias.data.zero_()
        else:
            raise KeyError(f"Unknown distribution {self.distribution}")


class BertCLSPooler(BertPoolerBase):
    def __init__(self, config):
        super().__init__(config)
        self.batch_id = 0
        self.pooler_dropout = False
        self.pooler_layer_norm = False

    def forward(self, hidden_states, attention_mask=None):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token ([CLS])
        token_tensor = hidden_states[:, 0]

        # Save token_tensor to disk
        # token_tensor_np = token_tensor.detach().cpu().numpy()
        # filename = '/logfiles/dodge-et-al-2020/' + f'token_tensor_{self.batch_id}.npy'
        # np.save(filename, token_tensor_np, allow_pickle=True)
        # self.batch_id += 1

        # RoBERTa uses an additional dropout here (before the linear transformation)
        if self.pooler_dropout:
            token_tensor_dropout = self.dropout(token_tensor)  # this we don't have in default BertPooler
        else:
            token_tensor_dropout = token_tensor

        pooled_linear_transform = self.dense(token_tensor_dropout)

        if self.pooler_layer_norm:  # apply LayerNorm to tanh pre-activations
            normalized_pooled_linear_transform = self.LayerNorm(pooled_linear_transform)
        else:
            normalized_pooled_linear_transform = pooled_linear_transform

        pooled_activation = self.pooler_activation(normalized_pooled_linear_transform)

        return pooled_activation, normalized_pooled_linear_transform, pooled_linear_transform, token_tensor


class BertModelWithPooler(BertModel):
    def __init__(self, config):
        super().__init__(config)
        self.pooler = BertCLSPooler(config)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
    ):
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            # Provided a padding mask of dimensions [batch_size, seq_length]
            # - if the model is a decoder, apply a causal mask in addition to the padding mask
            # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
            if self.config.is_decoder:
                batch_size, seq_length = input_shape
                seq_ids = torch.arange(seq_length, device=device)
                causal_mask = seq_ids[None, None, :].repeat(batch_size, seq_length, 1) <= seq_ids[None, :, None]
                causal_mask = causal_mask.to(
                    attention_mask.dtype
                )  # causal and attention masks must have same type with pytorch version < 1.3
                extended_attention_mask = causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
            else:
                extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                "Wrong shape for input_ids (shape {}) or attention_mask (shape {})".format(
                    input_shape, attention_mask.shape
                )
            )

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # If a 2D ou 3D attention mask is provided for the cross-attention
        # we need to make broadcastabe to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)

            if encoder_attention_mask.dim() == 3:
                encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
            elif encoder_attention_mask.dim() == 2:
                encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]
            else:
                raise ValueError(
                    "Wrong shape for encoder_hidden_shape (shape {}) or encoder_attention_mask (shape {})".format(
                        encoder_hidden_shape, encoder_attention_mask.shape
                    )
                )

            encoder_extended_attention_mask = encoder_extended_attention_mask.to(
                dtype=next(self.parameters()).dtype
            )  # fp16 compatibility
            encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * -10000.0
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = (
                    head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
                )  # We can specify head_mask for each layer
            head_mask = head_mask.to(
                dtype=next(self.parameters()).dtype
            )  # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.config.num_hidden_layers

        embedding_output = self.embeddings(
            input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
        )
        sequence_output = encoder_outputs[0]

        pooled_activation, normalized_pooled_linear_transform, pooled_linear_transform, token_tensor = self.pooler(sequence_output, attention_mask)
        # pooled activation is the results of the applying the pooler's tanh
        # pooler_output is the input to the pooler's tanh
        # token_tensor is the pooled vector, either CLS, 5th token, or mean over tokens

        outputs = (sequence_output, pooled_activation, normalized_pooled_linear_transform, pooled_linear_transform, token_tensor) + encoder_outputs[1:]  # add hidden_states and attentions if they are here

        return outputs  # sequence_output, pooled_activation, pooled_linear_transform, token_tensor, (hidden_states), (attentions)


class BertForSequenceClassificationWithPooler(BertForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModelWithPooler(config)
        self.dropout = nn.Dropout(0.1)
        # self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)
        self.classifier = nn.Linear(config.hidden_size, 1)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        pooled_activation, normalized_pooled_linear_transform, pooled_linear_transform, token_tensor = outputs[1:5]  # get outputs from the pooler
        pooled_activation_dropout = self.dropout(pooled_activation)
        logits = self.classifier(pooled_activation_dropout)

        outputs = (logits, pooled_activation_dropout, pooled_activation, normalized_pooled_linear_transform, pooled_linear_transform, token_tensor) + outputs[5:]  # add hidden states and attention if they are here

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits.view(-1), labels)

            outputs = (loss,) + outputs

        return outputs

    def manually_init_weights(self, std):
        # Initialize weights following: https://arxiv.org/abs/2002.06305
        print(f'Initializing weights of linear classifier: mean = 0.0, std = {std}')
        self.classifier.weight.data.normal_(mean=0.0, std=std)
        self.classifier.bias.data.zero_()
        # self.classifier.bias.data.normal_(mean=0.0, std=std)
