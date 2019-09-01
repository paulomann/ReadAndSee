import numpy as np
import torch
from collections import defaultdict
from readorsee.data.models import Config


class SIF():

    def __init__(self):
        pass

    def get_weighted_average(self, x, masks, sif_weights):
        assert sif_weights.size(0) == x.size(0)
        assert sif_weights.size(1) == x.size(1)
        assert sif_weights.size(0) == masks.size(0)
        assert sif_weights.size(1) == masks.size(1)
        sif_embeddings = torch.zeros((x.size(0), x.size(2)),
                         device=sif_weights.device, dtype=sif_weights.dtype)
        for i in range(x.size(0)):
            sif_embeddings[i] = sif_weights[i] @ x[i]
        
        sif_embeddings = sif_embeddings.div(masks.sum(dim=1).view(-1, 1))
        return sif_embeddings

    def remove_pc(self, sif_embeddings):
        """
        Remove the projection on the principal components
        :param sif_embeddings: Is the data containing the weighted sif embedding
        :return: is the data after removing its projection
        """
        _, _, V = sif_embeddings.svd()
        V = -V[:, 0]
        # Remove th first singular vector
        sif_embeddings = (sif_embeddings - (sif_embeddings @ V).view(-1, 1) * V)
        return sif_embeddings

    def SIF_embedding(self, embeddings, masks, sif_weights):
        """
        Compute the scores between pairs of sentences using weighted average + 
        removing the projection on the first principal component
        :param embeddings: tensor of size (A, B, C), where A is the number
                           of sentences (batch), B is the length of the biggest
                           sentence in sentences, and C is the embedding size
        :param masks: tensor of size (A, B), exactly the same size as embeddings
                      param, where masks[i,j] is 1 if a word j is in the sentece
                      i, and 0 otherwise
        :param sif_weights: tensor of size (A, B), exactly the same size as the
                            masks param, but instead of 1 and 0, it contains the
                            SIF weight for each word in the sentence, e.g.,
                            sif_weights[i, :] contains the weights of words for
                            the sentence i
        :return: emb, emb[i, :] is the embedding for sentence i
        """

        emb = self.get_weighted_average(embeddings, masks, sif_weights)
        emb = self.remove_pc(emb)
        return emb

    @staticmethod   
    def get_SIF_weights(sentences, a=1e-3):
        """
        :return:    tensor of size (A, B), exactly the same size as the
                    masks param of SIF_embedding function, but instead
                    of 1 and 0, it contain the SIF weight for each word
                    in the sentence, e.g., sif_weights[i, :] contains
                    the weights of words for the sentence i.
        """
        word_freqs = defaultdict(int)
        vocab_size = 0
        word2weight = {}
        max_sent_size = 0
        for sent in sentences:
            if len(sent) > max_sent_size:
                    max_sent_size = len(sent)
            for token in sent:
                word_freqs[token] += 1
        
        for k, v in word_freqs.items():
            vocab_size += v
        for k, v in word_freqs.items():
            word2weight[k] = a / (a + v/vocab_size)

        sif = []
        for sent in sentences:
            sif_sent = [word2weight[token] for token in sent]
            sif_sent = torch.tensor(sif_sent)
            sent_size = sif_sent.size(0)
            sif_sent = torch.cat(
                [sif_sent, torch.zeros(max_sent_size - sent_size)])
            sif.append(sif_sent)
        
        sif = torch.stack(sif)

        return sif
    

class PMEAN():

    def __init__(self):
        self.operations = dict([
            ('mean', 
                lambda word_emb: torch.mean(word_emb, dim=0)),
            ('max',
                lambda word_emb: torch.max(word_emb, dim=0)[0]),
            ('min',
                lambda word_emb: torch.min(word_emb, dim=0)[0]),
            ('p_mean_2',
                lambda word_emb: [self.gen_mean(word_emb, p=2.0).real]),
            ('p_mean_3',
                lambda word_emb: [self.gen_mean(word_emb, p=3.0).real])
        ])
    
    def gen_mean(self, vals, p):
        p = float(p)
        return np.power(
            np.mean(
                np.power(
                    np.array(vals, dtype=complex),
                    p),
                axis=0),
            1 / p
        )

    def znorm(self, embeddings):
        # Equivalent to https://github.com/UKPLab/arxiv2018-xling-sentence-embeddings/issues/3
        # without the normalization part
        mean = embeddings.mean(dim=0)
        std = embeddings.std(dim=0)
        embeddings = (embeddings - mean).div(std)
        embeddings[torch.isnan(embeddings)] = 0.
        embeddings[torch.isinf(embeddings)] = 1.
        return embeddings

    def PMEAN_embedding(self, embeddings, masks,
                        chosen_operations=["mean","max","min"]):
        """
        :param embeddings: tensor of size (A, B, C), where A is the number
                           of sentences (batch), B is the length of the biggest
                           sentence in sentences, and C is the embedding size
        :param masks: tensor of size (A, B), exactly the same size as embeddings
                      param, where masks[i,j] is 1 if a word j is in the sentece
                      i, and 0 otherwise
        :param chosen_operations: operations to concatenate the power means
        """
        sent_embs = []
        for i in range(embeddings.size(0)):
            sent = torch.index_select(embeddings[i, ...], 0,
                                    masks[i, ...].nonzero().squeeze())
            op_embs = []
            for o in chosen_operations:
                emb = self.operations[o](sent)
                emb[torch.isnan(emb)] = 0
                emb[torch.isinf(emb)] = 1
                op_embs += [emb]
            sent_embs += [torch.cat(op_embs, dim=0)]

        embeddings = torch.stack(sent_embs)
        embeddings = self.znorm(embeddings)
        return embeddings


def get_mean(x, masks):
    config = Config.getInstance()
    media_config = getattr(config, config.general["media_type"])

    if media_config["mean"] == "pmean":
        pmean = PMEAN()
        means = media_config["pmean"]
        pmean_embedding = pmean.PMEAN_embedding(x, masks, means)
        return pmean_embedding

    elif media_config["mean"] == "avg":
        x = x.sum(dim=1)
        masks = masks.sum(dim=1).view(-1, 1).float()
        x = torch.div(x, masks)
        x[torch.isnan(x)] = 0
        x[torch.isinf(x)] = 1
        return x
    else:
        raise NotImplementedError(f"There is no {media_config['mean']} aggregator.")