import re

import jieba
import jieba.analyse


class MyNLP(object):
    def __init__(self, USER_DICT=None, STOP_WORDS_PATH=None):
        if USER_DICT:
            jieba.load_userdict(USER_DICT)
        self.STOP_WORDS_PATH = STOP_WORDS_PATH
        self.stop_words = self._stop_words

    def get_pure_corpus(self, sentence, returnstr=True):
        segment = jieba.cut(re.sub('[^0-9a-zA-Z\u4e00-\u9fa5]+', ' ', sentence.strip().lower()))
        if returnstr:
            s = ' '.join(filter(lambda x: x not in self.stop_words, segment))
        else:
            s = list(filter(lambda x: x not in self.stop_words, segment))
        return s

    def get_key_words(self,
                      sentence,
                      allowPOS=['v', 'vg', 'vd', 'vn', 'n', 'nr', 'nr1', 'nr2', 'nrj', 'nrf', 'ns', 'nsf', 'nt', 'nz',
                                'nl', 'ng'],
                      topK=300):
        params = {'sentence': sentence, 'topK': topK, 'allowPOS': allowPOS}
        key_words = set(jieba.analyse.tfidf(**params) + jieba.analyse.textrank(**params))
        return list(key_words)

    @property
    def _stop_words(self):
        if self.STOP_WORDS_PATH:
            with open(self.STOP_WORDS_PATH) as f:
                stop_words = [line.strip() for line in f.readlines()] + [' ']
        else:
            stop_words = ['']
        return stop_words

# USER_DICT = '/DATA/UserDict/finWordDict.txt'
# STOP_WORDS_PATH = "/DATA/UserDict/stop_words.txt"
# mynlp = MyNLP(USER_DICT, STOP_WORDS_PATH)
