```python
class SimpleDoc2Vec(object):

    def __init__(self, sentence, tokenizer, word2vec):
        self.word2vec = word2vec
        self.words = tokenizer(sentence)
        self.vectors = np.array([self.word2vec[w] for w in self.words if w in self.word2vec])

    def doc2vec_1(self):
        return self.vectors.mean(0)

    def doc2vec_2(self):
        v = self.vectors.sum(0)
        return v / np.sqrt((v ** 2).sum())
```
