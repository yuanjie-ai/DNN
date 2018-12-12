```python
class SimpleDoc2Vec(object):

    def __init__(self, tokenizer, word2vec):
        self.word2vec = word2vec
        self.tokenizer = tokenizer

    def doc2vec_1(self, sentence):
        words = self.tokenizer(sentence)
        return self._get_vet(words).mean(0)

    def doc2vec_2(self, sentence):
        words = self.tokenizer(sentence)
        v = self._get_vet(words).sum(0)
        return v / np.sqrt((v ** 2).sum())

    def _get_vet(self, words):
        return np.array([self.word2vec[w] for w in words if w in self.word2vec])
```
