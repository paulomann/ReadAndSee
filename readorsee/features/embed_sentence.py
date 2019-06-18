import numpy as np
from sklearn.decomposition import TruncatedSVD


class SIF():

    def __init__(self):
        pass

    def get_weighted_average(self, We, x, w):
        """
        Compute the weighted average vectors
        :param We: We[i,:] is the vector for word i
        :param x: x[i, :] are the indices of the words in sentence i
        :param w: w[i, :] are the weights for the words in sentence i
        :return: emb[i, :] are the weighted average vector for sentence i
        """
        n_samples = x.shape[0]
        emb = np.zeros((n_samples, We.shape[1]))
        for i in range(n_samples):
            emb[i,:] = w[i,:].dot(We[x[i,:],:]) / np.count_nonzero(w[i,:])
        return emb

    def compute_pc(self, X, npc=1):
        """
        Compute the principal components. DO NOT MAKE THE DATA ZERO MEAN!
        :param X: X[i,:] is a data point
        :param npc: number of principal components to remove
        :return: component_[i,:] is the i-th pc
        """
        svd = TruncatedSVD(n_components=npc, n_iter=7, random_state=0)
        svd.fit(X)
        return svd.components_

    def remove_pc(self, X, npc=1):
        """
        Remove the projection on the principal components
        :param X: X[i,:] is a data point
        :param npc: number of principal components to remove
        :return: XX[i, :] is the data point after removing its projection
        """
        pc = self.compute_pc(X, npc)
        if npc==1:
            XX = X - X.dot(pc.transpose()) * pc
        else:
            XX = X - X.dot(pc.transpose()).dot(pc)
        return XX

    def SIF_embedding(self, We, x, w):#, params):
        """
        Compute the scores between pairs of sentences using weighted average + 
        removing the projection on the first principal component
        :param We: We[i,:] is the vector for word i
        :param x: x[i, :] are the indices of the words in the i-th sentence
        :param w: w[i, :] are the weights for the words in the i-th sentence
        :param params.rmpc: if >0, remove the projections of the sentence 
                            embeddings to their first principal component
        :return: emb, emb[i, :] is the embedding for sentence i
        """
        emb = self.get_weighted_average(We, x, w)
        emb = self.remove_pc(emb, 1)
        return emb
    

class PMEAN():

    def __init__(self):
        pass