- C
```python
def my_glove(CORPUS_PATH, GLOVE_HOME='/Users/yuanjie/GitHub/glove/build'):
    import os
    from pathlib import Path
    _path = Path(CORPUS_PATH).parent
    _vocab = "%s/vocab.txt" % _path
    _cooccur = "%s/cooccur.bin" % _path
    _shuf = "%s/cooccur.shuf.bin" %_path
    _vectors = "%s/vectors" %_path
    
    cmd_vocab = "%s/vocab_count -min-count 128 < %s > %s" % (GLOVE_HOME, CORPUS_PATH, _vocab)
    cmd_cooccur = "%s/cooccur -window-size 15 -vocab-file %s -memory 16 < %s > %s" % (GLOVE_HOME, _vocab, CORPUS_PATH, _cooccur)
    cmd_shuffle = "%s/shuffle -memory 16 < %s > %s" % (GLOVE_HOME, _cooccur, _shuf)
    cmd_glove = "%s/glove -vector-size 100 -threads 32 -alpha 0.75 -x-max 100.0 -eta 0.05 -binary 2 -save-file %s" % (GLOVE_HOME, _vectors)
    cmd = ' && '.join([cmd_vocab, cmd_cooccur, cmd_shuffle, cmd_glove])
    os.system(cmd)
    print(os.popen('cd %s && ls -l' % _path).read()) # 词向量与数据同目录
```



- Python
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
                yield i.strip().lower().split(' ')
```
