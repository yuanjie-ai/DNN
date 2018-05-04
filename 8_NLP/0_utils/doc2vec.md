```python
import datetime
from pathlib import Path

from gensim.models.doc2vec import Doc2Vec, TaggedDocument, TaggedLineDocument
from tqdm import tqdm


class MyDoc2Vec(Doc2Vec):
    def __init__(self, corpus, **kwargs):
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
        model = MyDoc2Vec(docs)
        model.doc2vec()
        """
        self.corpus = corpus
        self.corpus_convert()

    def doc2vec(self,
                vector_size=300,
                window=10,
                min_count=1,
                dm=1,
                hs=0,
                negative=5,
                epochs=10):
        """
        :param size: Dimensionality of the feature vectors.
        :param window: The maximum distance between the current and predicted word within a sentence.
        :param min_count: Ignores all words with total frequency lower than this.
        :param dm: Defines the training algorithm. If `dm=1`, 'distributed memory' (PV-DM) is used.
            Otherwise, `distributed bag of words` (PV-DBOW) is employed.
        :param hs: int {1,0} shared_softmax
            If 1, hierarchical softmax will be used for model training.
            If set to 0, and `negative` is non-zero, negative sampling will be used.
        :param negative:
            If > 0, negative sampling will be used, the int for negative specifies how many "noise words"
                should be drawn (usually between 5-20).
            If set to 0, no negative sampling is used.
        """
        model = Doc2Vec(documents=tqdm(self.corpus), vector_size=vector_size, window=window, min_count=min_count, dm=dm,
                        hs=hs, negative=negative, epochs=epochs, workers=32)

        return model

    def corpus_convert(self):
        if isinstance(self.corpus, str) and Path(self.corpus).is_file():
            self.corpus = TaggedLineDocument(self.corpus)
        else:
            self.corpus = [TaggedDocument(line, [idx]) for idx, line in enumerate(self.corpus)]

    def model_save(self, model, path=None):
        if path:
            model.save(path)
        else:
            model.save('./%s___%s.model' % (str(datetime.datetime.today())[:22], model.__str__()))

# class MyDoc2Vec(Doc2Vec):
#     def __init__(self, corpus, **kwargs):
#
#         self.corpus(corpus)
#         super().__init__(documents=self.corpus, **kwargs)  # documents=self.documents
#
#     def corpus(self, corpus):
#         if isinstance(self.corpus, str) and Path(corpus).is_file():
#             self.corpus = tqdm(TaggedLineDocument(corpus), desc='TaggedLineDocument')
#         else:
#             self.corpus = tqdm(self.CorpusIter(corpus), desc='TaggedDocument')
#
#     class CorpusIter(object):
#         def __init__(self, corpus):
#             self.corpus = corpus
#
#         def __iter__(self):
#             for idx, line in enumerate(self.corpus):
#                 yield TaggedDocument(line.split(), [idx])
#
#                 # @property
#                 # def _corpus(self):
#                 #     """
#                 #     ['Well done!', 'Good work']
#                 #     """""
```
