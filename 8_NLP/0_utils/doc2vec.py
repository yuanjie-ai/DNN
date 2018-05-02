from pathlib import Path

from gensim.models.doc2vec import Doc2Vec, TaggedDocument, TaggedLineDocument
from tqdm import *


class MyDoc2Vec(Doc2Vec):
    def __init__(self, corpus, **kwargs):

        self.corpus(corpus)
        super().__init__(documents=self.corpus, **kwargs) # documents=self.documents

    def corpus(self, corpus):
        if isinstance(self.corpus, str) and Path(corpus).is_file():
            self.corpus = tqdm(TaggedLineDocument(corpus), desc='TaggedLineDocument')
        else:
            self.corpus = tqdm(self.CorpusIter(corpus), desc='TaggedDocument')

    class CorpusIter(object):
        def __init__(self, corpus):
            self.corpus = corpus

        def __iter__(self):
            for idx, line in enumerate(self.corpus):
                yield TaggedDocument(line.split(), [idx])
                
    # @property
    # def _corpus(self):
    #     """
    #     ['Well done!', 'Good work']
    #     """""
    #     for idx, line in enumerate(self.corpus):
    #         yield TaggedDocument(line.split(), [idx])
