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
from pathlib import Path

from gensim.corpora import Dictionary
from gensim.models import LdaMulticore
from gensim.models.word2vec import LineSentence
from tqdm import tqdm


class MyLDA(object):
    """
    docs = [['Well', 'done!'],
             ['Good', 'work'],
             ['Great', 'effort'],
             ['nice', 'work'],
             ['Excellent!'],
             ['Weak'],
             ['Poor', 'effort!'],
             ['not', 'good'],
             ['poor', 'work'],
             ['Could', 'have', 'done', 'better.']]
    mylda = MyLDA(docs)
    lda = mylda.lda()
    lda[mylda.dict_.doc2bow(['Poor'])]
    """

    def __init__(self, corpus):
        """
        :param corpus: [['w1', 'w2']] or file_path
        """
        self.corpus = corpus
        self.__corpus_convert()
        self.word2id = self.dict_.token2id
        self.id2word = self.dict_  # dict(self.dict_)

    def lda(self, num_topics=10, iterations=50, seed=None):
        lda = LdaMulticore(self.corpus,
                           id2word=self.dict_,
                           num_topics=num_topics,
                           iterations=iterations,
                           random_state=seed,
                           batch=False,
                           workers=32)
        return lda

    def get_topic_terms(self, lda_model, topicid, topn=None):
        """
        :param lda_model:
        :param topicid:
        :param topn:
        :return: 获取某个主题下的前topn个词语
        """
        if topn:
            pass
        else:
            topn = lda_model.num_terms
        res = lda_model.get_topic_terms(topicid, topn)
        return self.__id2word(res)

    def get_term_topics(self, lda_model, word, minimum_probability=0):
        """
        :param lda_model:
        :param word: id or str
        :param minimum_probability:
        :return: 只会获得在整个样本下，某个词属于某些主题的可能性，而并不是针对特定文档的某个词属于某些主题的可能性
        """
        if isinstance(word, str):
            word = self.word2id[word]
        res = lda_model.get_term_topics(word, minimum_probability)
        return self.__id2word(res)

    def get_document_topics(self, lda_model, bow, minimum_probability=None, minimum_phi_value=None,
                            per_word_topics=False):
        """
        :param lda_model:
        :param bow:
        :param minimum_probability: 确定主题的阈值
        :param minimum_phi_value: 判断词属于主题的阈值
        :param per_word_topics: False: 获取某个文档最有可能具有的主题列表(default)
         True:
            ([a],[b],[c])
            第一行为a列表，显示的内容是可能性大于minimum_probability的主题及其可能性 # per_word_topics=False仅显示该行
            第二行为b列表，显示每个词及其可能属于的主题，且其可能性大于minimum_phi_value
            第三行为c列表，显示每个词及其可能属于的主题以及可能性
        :return: 类似lda_model.__getitem__(bow, eps=None) or lda_model[bow]
        """
        res = lda_model.get_document_topics(bow, minimum_probability, minimum_phi_value, per_word_topics)
        return res

    def __corpus_convert(self):
        if isinstance(self.corpus, str) and Path(self.corpus).is_file():
            self.corpus = tqdm(LineSentence(self.corpus), desc='Processing')
        self.dict_ = Dictionary(self.corpus)
        self.corpus = [self.dict_.doc2bow(doc) for doc in tqdm(self.corpus, desc='Bag Of Words')]

    def __id2word(self, res=[(0, 888)]):
        return sorted([(self.id2word[idx], v) for idx, v in res], key=lambda x: -x[1])

    def update(self):
        """
        >>> lda = LdaMulticore(corpus, num_topics=10)
        You can then infer topic distributions on new, unseen documents, with
        >>> doc_lda = lda[doc_bow]

        The model can be updated (trained) with new documents via

        >>> lda.update(other_corpus)
        """
        pass

```


---
[1]: https://blog.csdn.net/tiffanyrabbit/article/details/76445909
