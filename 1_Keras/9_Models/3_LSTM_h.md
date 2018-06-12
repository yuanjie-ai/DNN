```python
import keras.backend as K
from keras.layers import Dense, Dropout, Embedding, LSTM, Lambda
from keras.models import Input, Model

main_input = Input(shape=(None,))
x = Embedding(10000, 128, mask_zero=True)(main_input)  # 用了mask_zero，填充部分自动为0

#######################################################################
lstm = LSTM(128, return_sequences=True, return_state=True)(x)  # 返回一个list: [return_sequences, return_state, ...]
# lstm[0]就是lstm的状态向量序列，先补充一个0向量（h_0）
lstm_sequence = Lambda(lambda x: K.concatenate([K.zeros_like(x[0])[:, :1], x[0]], 1))(lstm)
# lstm[1]就是lstm最后的状态
lstm_state = Lambda(lambda x: x[1])(lstm)
#######################################################################

# 分类器
x = Dropout(0.5)(lstm_state)
main_out = Dense(1, activation='sigmoid')(x)

# 对最后状态的影响程度 ∥hn−hi∥−∥hn−hi+1∥
aux_out = Lambda(lambda x: K.sqrt(K.sum((x[0] - K.expand_dims(x[1], 1)) ** 2, 2) / K.sum(
    x[1] ** 2, 1, keepdims=True)))([lstm_sequence, lstm_state])

model = Model(inputs=main_input, outputs=main_out)  # 文本情感分类模型
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model_dist = Model(inputs=main_input, outputs=aux_out)  # 计算权重的模型
model_dist.compile(loss='mse',
                   optimizer='adam')

```

## Demo
```python
import numpy as np
import pandas as pd
import jieba
from tqdm import tqdm
tqdm.pandas(tqdm)


pos = pd.read_excel('pos.xls', names=['s'], header=None).assign(label=1)
neg = pd.read_excel('neg.xls', names=['s'], header=None).assign(label=0)
df = pd.concat([pos, neg], ignore_index=True)
df['words'] = df.s.progress_apply(jieba.lcut)

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
        doc2num = []
        for text in tqdm(self.X, desc='Doc To Number'):
            s = [self.word2idx.get(i, 0) for i in text[:self.maxlen]]
            doc2num.append(s + [0]*(self.maxlen-len(s)))  # 未登录词全部用0表示
        self.doc2num = doc2num
        
bow = BOW(df.words)
_X, _y = np.asarray(bow.doc2num), df.label.values.reshape(-1, 1)

# 打乱数据
idx = list(range(len(_X)))
np.random.seed(42)
np.random.shuffle(idx)
_X = _X[idx]
_y = _y[idx]

model.fit(_X, _y, batch_size=256, epochs=3, validation_split=0.25)

# 词输入重要性
def saliency(s): #简单的按saliency排序输出的函数
    ws = jieba.lcut(s)
    x_ = np.array([[bow.word2idx.get(w,0) for w in ws]])
    score = np.diff(model_dist.predict(x_)[0])
    idxs = score.argsort()
    return [(i,ws[i],-score[i]) for i in idxs]

saliency('用过了才来评价，挺好用的和我在商场买的一样，应该是正品，烧水也快。五分')
"""
[(18, '正品', 0.22221938),
 (6, '挺好用', 0.20402634),
 (24, '五分', 0.11586456),
 (11, '商场', 0.09979665),
 (0, '用过', 0.0705657),
 (8, '和', 0.057504237),
 (20, '烧水', 0.051930293),
 (14, '一样', 0.04657948),
 (22, '快', 0.042410836),
 (23, '。', 0.042060807),
 (10, '在', 0.038769543),
 (19, '，', 0.03445798),
 (16, '应该', 0.031840503),
 (7, '的', 0.03173977),
 (9, '我', 0.031158566),
 (3, '来', 0.028542519),
 (15, '，', 0.025797188),
 (13, '的', 0.020577371),
 (21, '也', 0.0187037),
 (17, '是', 0.008114278),
 (5, '，', 0.0049973726),
 (1, '了', 0.0024805665),
 (12, '买', -0.030952454),
 (4, '评价', -0.09881866),
 (2, '才', -0.10036659)]
 
效果可见一斑，排在前面的词语基本上是情感倾向比较强烈的词语。值得指出的是，这种重要性的评估方案还会自动地考虑词语的位置所造成的影响，假如一个情感词在句子中重复出现，那么后出现的词语一般来说权重会更低（因为前面的已经能让我们完成分类了，后面的权重就下降了）
"""
```
