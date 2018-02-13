```python
import jieba
corpus = ["我 来到 北京 清华大学",  # 第一类文本切词后的结果，词之间以空格隔开
          "我 爱 北京 天安门"]  # 第二类文本的切词结果
```

- `CountVectorizer`
```python
from sklearn.feature_extraction.text import CountVectorizer

countVector=CountVectorizer()
c = countVector.fit_transform(corpus)
c.todense()
countVector.get_feature_names()

# matrix([[1, 0, 1, 1],
#         [1, 1, 0, 0]], dtype=int64)
# ['北京', '天安门', '来到', '清华大学']
```

- `TfidfVectorizer`
```python
from sklearn.feature_extraction.text import TfidfVectorizer

tfidfVector = TfidfVectorizer()
tfidfVector.fit_transform(text).todense()
tfidfVector.get_feature_names()

# matrix([[0.44943642, 0.        , 0.6316672 , 0.6316672 ],
#         [0.57973867, 0.81480247, 0.        , 0.        ]])
# ['北京', '天安门', '来到', '清华大学']
```
