```python
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer

sentences = [["cat", "say", "meow"], ["dog", "say", "woof"]]
corpus = [' '.join(i) for i in sentences]
tfidfVector = TfidfVectorizer()
tfidf_vector = tfidfVector.fit_transform(corpus)


model = Word2Vec(sentences, min_count=1, size=3) # min_count=1保证
word_vector = model.wv.__getitem__(tfidfVector.get_feature_names()) # 词向量与tfidf顺序一致 model.wv.index2entity
tfidf_vector*word_vector # 句向量：词向量进行tfidf加权之和（glove模长代表词重要性无需tfidf加权）
```
