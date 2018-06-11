from tqdm import tqdm


class BOW(object):
    def __init__(self, X, min_count=10, maxlen=100):
        self.X = X
        self.min_count = min_count
        self.maxlen = maxlen
        self.__word_count()
        self.__idx()
        self.__doc2num()

    def __word_count(self):
        wc = {}
        for ws in tqdm(self.X, desc='   Word Count'):
            for w in ws:
                if w in wc:
                    wc[w] += 1
                else:
                    wc[w] = 1
        self.word_count = {i: j for i, j in wc.items() if j >= self.min_count}

    def __idx(self):
        self.idx2word = {i + 1: j for i, j in enumerate(self.word_count)}
        self.word2idx = {j: i for i, j in self.idx2word.items()}

    def __doc2num(self):
        texts = []
        for text in tqdm(self.X, desc='Doc To Number'):
            texts.append(
                [self.word2idx.get(i, 0) for i in text[:self.maxlen]] + [0] * (self.maxlen - len(text)))  # 未登录词全部用0表示
        self.doc2num = doc2num
