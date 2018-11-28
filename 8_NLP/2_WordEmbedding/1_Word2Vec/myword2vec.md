```python
import datetime
from pathlib import Path

from gensim.models.word2vec import LineSentence, Word2Vec
from tqdm import tqdm


class MyWord2Vec(object):
    """
    架构：skip-gram（慢、对罕见字有利）vs CBOW（快）
    训练算法：分层softmax（对罕见字有利）vs 负采样（对常见词和低纬向量有利）
    负例采样准确率提高，速度会慢，不使用negative sampling的word2vec本身非常快，但是准确性并不高
    欠采样频繁词：可以提高结果的准确性和速度（适用范围1e-3到1e-5）
    文本（window）大小：skip-gram通常在10附近，CBOW通常在5附近
    size: n = sqrt(词汇量)/2
    """

    def __init__(self, corpus=None):
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
        model = MyWord2Vec(docs)
        model.word2vec()
        
        model.init_sims(replace=True) # 对model进行锁定，并且据说是预载了相似度矩阵能够提高后面的查询速度，但是你的model从此以后就read only了
        """
        self.corpus = corpus
        self.corpus_convert()

    def word2vec(self,
                 vector_size=300,
                 window=5,
                 min_count=1,
                 sg=0,
                 hs=0,
                 negative=5,
                 epochs=10):
        """
        :param vector_size:  Dimensionality of the feature vectors.
        :param window: The maximum distance between the current and predicted word within a sentence.
        :param min_count: Ignores all words with total frequency lower than this.
        :param sg: int {1, 0}
            Defines the training algorithm. If 1, skip-gram is employed; otherwise, CBOW is used.
        :param hs: int {1,0} shared_softmax
            If 1, hierarchical softmax will be used for model training.
            If set to 0, and `negative` is non-zero, negative sampling will be used.
        :param negative:
            If > 0, negative sampling will be used, the int for negative specifies how many "noise words"
            should be drawn (usually between 5-20).
            If set to 0, no negative sampling is used.
        """
        model = Word2Vec(tqdm(self.corpus, desc="Word2Vec Preprocessing"), size=vector_size, window=window, min_count=min_count, sg=sg, hs=hs,
                         negative=negative, iter=epochs, workers=32)
        return model

    def corpus_convert(self):
        if isinstance(self.corpus, str) and Path(self.corpus).is_file():
            self.corpus = LineSentence(self.corpus)

    def model_save(self, model, path=None):
        if path:
            model.save(path)
        else:
            model.save('./%s___%s.model' % (str(datetime.datetime.today())[:22], model.__str__()))
```
