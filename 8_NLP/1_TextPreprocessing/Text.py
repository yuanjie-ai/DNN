import jieba
from snownlp import SnowNLP
class Text(object):
    def __init__(self):
        pass

    @staticmethod
    def get_text_tokens(text, stop_words_path=None):
        text_tokens = jieba.cut(text.strip())
        stop_words = Text.get_stop_words(stop_words_path)
        word_list = []
        for word in text_tokens:
            if word not in stop_words:
                if word != '\t':
                    word_list.append(word)
        return word_list # 词频Counter(word_list)

    @staticmethod
    def get_stop_words(path):
        with open(path) as f:
            stop_words = [line.strip() for line in f.readlines()]
        return stop_words

    @staticmethod
    def fan2jian(text):
        return SnowNLP(text).han

    @staticmethod
    def get_pinyin(text):
        return SnowNLP(text).pinyin
