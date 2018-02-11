```python
s = "江州市长江大桥参加了长江大桥的通车仪式"
text = Text.get_text_tokens(s, stop_words_path='./stop_words.txt') # 已去除停顿词
stop_words = Text.get_stop_words('./stop_words.txt')

countVector = CountVectorizer(stop_words=stop_words)
countVector.fit_transform(text).todense()
countVector.get_feature_names()
# matrix([[0, 0, 1, 0, 0],
#         [0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 1],
#         [0, 1, 0, 0, 0],
#         [0, 0, 0, 0, 1],
#         [0, 0, 0, 1, 0],
#         [1, 0, 0, 0, 0]], dtype=int64)
# Out[381]:
# ['仪式', '参加', '江州', '通车', '长江大桥']

tfidfVector = TfidfVectorizer(stop_words=stop_words)
tfidfVector.fit_transform(text).todense()
tfidfVector.get_feature_names()
# matrix([[0., 0., 1., 0., 0.],
#         [0., 0., 0., 0., 0.],
#         [0., 0., 0., 0., 1.],
#         [0., 1., 0., 0., 0.],
#         [0., 0., 0., 0., 1.],
#         [0., 0., 0., 1., 0.],
#         [1., 0., 0., 0., 0.]])
# Out[383]:
# ['仪式', '参加', '江州', '通车', '长江大桥']
```
