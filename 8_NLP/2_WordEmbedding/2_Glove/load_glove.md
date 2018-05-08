```python
#! -*- coding: utf-8 -*-

import numpy as np

class Simpler_Golve:
    def __init__(self, glove_file):
        embeddings = []
        self.words = []
        self.word2id = {}
        idx = 0
        for l in open(glove_file, encoding='utf-8'):
            _ = l.strip('\n').split(' ')
            if len(_) > 2:
                self.word2id[_[0]] = idx
                self.words.append(_[0])
                embeddings.append(_[1:])
                idx += 1
        self.embeddings = np.array(embeddings).astype(float)
        self.word_size = self.embeddings.shape[1]
        self.id2word = {j:i for i,j in self.word2id.items()}
        self.init_something()
    def init_something(self):
        self.normalized_embeddings = self.noramlize(self.embeddings)
        self.nb_context_words = None
    def noramlize(self, x):
        if len(x.shape) > 1:
            return x/np.clip(x**2, 1e-12, None).sum(axis=1).reshape((-1,1)+x.shape[2:])**0.5
        else:
            return x/np.clip(x**2, 1e-12, None).sum()**0.5
    def most_correlative(self, word, topn=20, normalize=True):
        if normalize:
            word_vec = self.normalized_embeddings[self.word2id[word]]
            word_sim = np.dot(self.normalized_embeddings, word_vec)
        else:
            word_vec = self.embeddings[self.word2id[word]]
            word_sim = np.dot(self.embeddings, word_vec)
        word_sim_sort = word_sim.argsort()[::-1]
        return [(self.id2word[i], word_sim[i]) for i in word_sim_sort[:topn]]
    def most_similar(self, word, topn=20, nb_context_words=100000):
        if nb_context_words != self.nb_context_words:
            embeddings_ = self.embeddings[:nb_context_words]
            embeddings_ = embeddings_ - embeddings_.mean(axis=0)
            U = np.dot(embeddings_.T, embeddings_)
            U = np.linalg.cholesky(U)
            embeddings_ = np.dot(self.embeddings, U)
            self.nb_context_words = nb_context_words
            self.normalized_embeddings_ = embeddings_/(embeddings_**2).sum(axis=1).reshape((-1,1))**0.5
        word_vec = self.normalized_embeddings_[self.word2id[word]]
        word_sim = np.dot(self.normalized_embeddings_, word_vec)
        word_sim_sort = word_sim.argsort()[::-1]
        return [(self.id2word[i], word_sim[i]) for i in word_sim_sort[1:topn]]
    def analogy(self, pos_word_1, pos_word_2, neg_word=None, topn=20):
        if neg_word:
            word_vec = self.embeddings[self.word2id[pos_word_1]] + \
                       self.embeddings[self.word2id[pos_word_2]] - \
                       self.embeddings[self.word2id[neg_word]]
        else:
            word_vec = self.embeddings[self.word2id[pos_word_1]] + \
                       self.embeddings[self.word2id[pos_word_2]]
        word_vec = word_vec/np.dot(word_vec, word_vec)**0.5
        word_sim = np.dot(self.normalized_embeddings, word_vec)
        word_sim_sort = word_sim.argsort()[::-1]
        return [(self.id2word[i], word_sim[i]) for i in word_sim_sort[:topn]]
    def sent2vec(self, sent, mode='sum'):
        idxs = [self.word2id[w] for w in sent if w in self.word2id]
        if idxs:
            if mode == 'sum':
                return self.embeddings[idxs].sum(axis=0)
            elif mode == 'avg':
                return self.embeddings[idxs].mean(axis=0)
            else:
                return np.zeros(self.word_size)
        else:
            return np.zeros(self.word_size)
    def keywords(self, sent, normalize=False):
        word_set = list(set([self.word2id[w] for w in sent if w in self.word2id]))
        word_vec = self.embeddings[word_set]
        sent_vec = self.sent2vec(sent)
        if normalize:
            sent_vec = self.noramlize(sent_vec)
            word_vec = self.normalized_embeddings[word_set]
        word_sim = np.dot(word_vec, sent_vec)
        word_sim_sort = word_sim.argsort()[::-1]
        return [(self.id2word[word_set[i]], word_sim[i]) for i in word_sim_sort]
    def sentence_similarity(self, sent_1, sent_2):
        sent_vec_1 = self.normalize(self.sent2vec(sent_1))
        sent_vec_2 = self.normalize(self.sent2vec(sent_2))
        return np.dot(sent_vec_1,sent_vec_2)
    def __getitem__(self, w):
        return self.embeddings[self.word2id[w]]


```
