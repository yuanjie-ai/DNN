- ModelUpdate
```python
from pathlib import Path

import gensim.models
from gensim.models.doc2vec import TaggedDocument, TaggedLineDocument
from gensim.models.word2vec import LineSentence
from tqdm import tqdm


class ModelUpdate(object):
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
    model = MyWord2Vec(docs).word2vec()
    ModelUpdate([['new', 'corpus']], model=model).train() # return model(updated)
    """

    def __init__(self, corpus, model, model_type='Word2Vec'):
        """
        :param corpus:
        :param model: model object or model file
        :param model_type: 'Word2Vec' or 'Doc2Vec'
        """
        self.corpus = corpus
        self.model = model
        self.model_type = model_type
        self.corpus_convert()

    def train(self):
        if isinstance(self.model, str) and Path(self.model).is_file():
            self.model = gensim.models.__getattribute__(self.model_type).load(self.model)
        self.model.build_vocab(self.corpus, update=True)
        self.model.train(tqdm(self.corpus), total_examples=self.model.corpus_count, epochs=self.model.epochs)
        return self.model  # inplace

    def corpus_convert(self):
        if self.model_type == 'Word2Vec':
            self.word2vec_corpus()
        else:
            self.doc2vec_corpus()

    def word2vec_corpus(self):
        if isinstance(self.corpus, str) and Path(self.corpus).is_file():
            self.corpus = LineSentence(self.corpus)

    def doc2vec_corpus(self):
        if isinstance(self.corpus, str) and Path(self.corpus).is_file():
            self.corpus = TaggedLineDocument(self.corpus)
        else:
            self.corpus = [TaggedDocument(line, [idx]) for idx, line in enumerate(self.corpus)]
```
