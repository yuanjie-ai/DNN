- [Sklearn LDA][1]
```python
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm


class MyLDA(object):
    """
    文章：主题加权
    主题：主题词加权
    """

    def __init__(self, corpus):
        self.corpus = corpus

    def lda(self, n_components):
        tf_model = CountVectorizer()
        tf_vec = tf_model.fit_transform(self.corpus) # 支持文件
        lda = LatentDirichletAllocation(n_components=n_components)
        lda.fit(tf_vec)
        return lda, tf_model.get_feature_names()

    @staticmethod
    def get_top_words(model, feature_names, topN=5):
        l = []
        for _, topic in tqdm(enumerate(model.components_)):
            l.append([feature_names[i] for i in topic.argsort()[:-(topN + 1):-1]])
        return l
```

- Gensim LDA
```python
```


---
[1]: https://blog.csdn.net/tiffanyrabbit/article/details/76445909
