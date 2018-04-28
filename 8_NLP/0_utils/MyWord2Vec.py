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
    """

    def __init__(self, corpus=None):
        """
        :param corpus: iterable 二维
        """
        self.corpus = corpus

    def word2vec(self,
                 size=300,
                 window=15,
                 min_count=1000,
                 workers=20,
                 sg=0,
                 hs=0,
                 negative=5,
                 iter=10):
        """
        :param sentences: iterable 二维
        :param size:  Dimensionality of the feature vectors.
        :param window: The maximum distance between the current and predicted word within a sentence.
        :param min_count: Ignores all words with total frequency lower than this.
        :param workers: njobs
        :param is_cbow: int {1, 0}
            Defines the training algorithm. If 1, CBOW is used, otherwise, skip-gram is employed.
        :param hs: int {1,0} shared_softmax
            If 1, hierarchical softmax will be used for model training.
            If set to 0, and `negative` is non-zero, negative sampling will be used.
        :param negative:
            If > 0, negative sampling will be used, the int for negative specifies how many "noise words"
            should be drawn (usually between 5-20).
            If set to 0, no negative sampling is used.
        :param iter:
        :param model_path:
        """
        model_train = lambda sentences: Word2Vec(sentences, size=size, window=window, min_count=min_count,
                                                 workers=workers, sg=sg, hs=hs, negative=negative, iter=iter)
        if isinstance(self.corpus, str) and Path(self.corpus).is_file():
            model = model_train(tqdm(LineSentence(self.corpus)))
        else:
            model = model_train(self.corpus)

        model.save('./%s___%s.model' % (str(datetime.datetime.today())[:22], model.__str__()))

        @staticmethod
        def model(self, model_path):
            """
            load Word2Vec model
            """
            return Word2Vec.load(model_path)
