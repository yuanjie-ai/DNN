```python
from pathlib import Path

from sklearn.feature_extraction import text
from tqdm import tqdm


class MyTfidf(object):
    def __init__(self, corpus):
        self.corpus = corpus

    @staticmethod
    def get_values(sentence, model):
        """
        :param sentence: ['w1', 'w2] (not iter)
        :param model: sklearn model
        :return: word rank
        """
        print(model.__class__)
        _dict = model.vocabulary_
        M = model.transform([' '.join(sentence)])
        return sorted([(M[:, _dict[i]].max(), i) for i in sentence if i in _dict])[::-1]

    def get_model(self, model='TfidfVectorizer'):
        """
        :param model: 'CountVectorizer' or 'TfidfVectorizer'
        :return:
        """
        _ = text.__getattribute__(model)()
        if isinstance(self.corpus, str) and Path(self.corpus).is_file():
            with open(self.corpus) as f:
                _.fit(tqdm(f, desc='Model Training'))
        else:
            _.fit(self.corpus)
        return _
```
