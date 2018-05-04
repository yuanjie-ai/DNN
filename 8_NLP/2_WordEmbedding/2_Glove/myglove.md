```python
from pathlib import Path
from glove import Corpus, Glove
from tqdm import tqdm


class MyGlove(object):
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
    model = MyGlove(docs).glove()
    """

    def __init__(self, corpus):
        self.corpus = corpus
        self.corpus_convert()

    def glove(self, no_components=30, learning_rate=0.05, alpha=0.75, max_count=100, max_loss=10.0, random_state=None):
        glove = Glove(no_components, learning_rate, alpha, max_count, max_loss, random_state)
        glove.fit(self.corpus)
        glove.add_dictionary(self.dictionary)
        return glove

    def corpus_convert(self):
        if isinstance(self.corpus, str) and Path(self.corpus).is_file():
            self.corpus = self.reader(self.corpus)
        corpus_model = Corpus()
        corpus_model.fit(tqdm(self.corpus, desc='Get Corpus'))
        self.corpus = corpus_model.matrix
        print("Non-zero elements: %d" % self.corpus.nnz)
        self.dictionary = corpus_model.dictionary

    def reader(self, file_path):
        with open(file_path) as f:
            for i in f:
                yield i.strip().split(' ')
```
