from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils.np_utils import to_categorical
from tqdm import tqdm


class TextPreprocessing(object):
    def __init__(self, X=None, y=None, maxlen=None, num_words=10 ** 5):
        self.maxlen = maxlen
        self.num_words = num_words
        self.__preprocessing(texts=tqdm(X, desc="Text Preprocessing"))
        if y is not None:
            self.y = to_categorical(y)

    def __preprocessing(self, texts):
        """
        :param texts: ['some thing to do', 'some thing to drink'] 与 sklearn 一致
        :return:
        """
        tokenizer = Tokenizer(self.num_words)  # 保留top num_words-1(词频降序)：最常见的num_words-1词
        tokenizer.fit_on_texts(texts)
        sequences = tokenizer.texts_to_sequences(texts)
        self.tokenizer = tokenizer
        self.word_index = tokenizer.word_index
        # self.word_counts = tokenizer.word_counts
        print(f"Get Unique Words: {len(self.word_index)}")
        self.X = pad_sequences(sequences, maxlen=self.maxlen, padding='post')
        if self.maxlen is None:
            self.maxlen = self.X.shape[1]

    def __label_mapper(self, labels):  # from collections import defaultdict
        # label2idx = defaultdict(int)
        label2idx = {}
        for i in labels:
            if i not in label2idx:
                label2idx[i] = len(label2idx)
        idx2label = {v: k for k, v in label2idx.items()}
        return label2idx, idx2label
