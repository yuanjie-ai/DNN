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
