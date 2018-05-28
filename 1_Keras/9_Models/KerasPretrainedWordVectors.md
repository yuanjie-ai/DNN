```python
import numpy as np
from keras.layers import Embedding
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils.np_utils import to_categorical
from tqdm import tqdm


class KerasPretrainedWordVectors(object):
    def __init__(self, X=None, y=None, word_vectors=None, max_num_words=10 ** 5):
        """
        :param X: ['Well done!', 'Good work']
        :param y:
        :param word_vectors: word vectors path
        :param max_num_words:
        """
        self.max_num_words = max_num_words
        
        if word_vectors is not None:
            self.__load_word_vectors(word_vectors)

        self.__preprocessing(texts=tqdm(X, desc="Text Preprocessing"))
        
        if y is not None:
            num_classes = len(set(y))
            if num_classes > 2:
                self.y = to_categorical(y, num_classes)
            else:
                self.y = y

    def embedding_layer(self):
        # prepare embedding matrix
        num_words = len(self.word_index) + 1  # 未出现的词0
        embedding_matrix = np.zeros((num_words, self.embeddings_dim))
        # embedding_matrix = np.random.random((num_words, EMBEDDING_DIM))
        # embedding_matrix = np.random.normal(size=(num_words, EMBEDDING_DIM))
        for word, idx in tqdm(self.word_index.items(), desc="Load Word Vectors Into EmbeddingLayer"):
            embedding_vector = self.embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[idx] = embedding_vector

        # note that we set trainable = False so as to keep the embeddings fixed
        embedding_layer = Embedding(num_words,
                                    self.embeddings_dim,
                                    weights=[embedding_matrix],
                                    input_length=self.max_sequence_length,
                                    trainable=False)
        return embedding_layer

    def __preprocessing(self, texts):
        """
        :param texts: ['some thing to do']
        :return:
        """
        tokenizer = Tokenizer(self.max_num_words)
        tokenizer.fit_on_texts(texts)
        sequences = tokenizer.texts_to_sequences(texts)
        self.max_sequence_length = max(map(len, sequences))
        self.tokenizer = tokenizer
        self.word_index = tokenizer.word_index
        print(f"Get Unique Words: {len(self.word_index)}")
        self.X = pad_sequences(sequences, maxlen=self.max_sequence_length, padding='post')

    def __load_word_vectors(self, word_vectors):
        embeddings_index = {}
        with open(word_vectors) as f:
            for line in tqdm(f, desc='Word Vectors Loading'):
                line = line.split()
                if len(line) > 2:
                    embeddings_index[line[0]] = np.asarray(line[1:], dtype='float32')
        self.embeddings_index = embeddings_index
        self.embeddings_dim = len(line[1:])

```
