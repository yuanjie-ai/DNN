1. 传递一个input_shape的关键字参数给第一层，input_shape是一个tuple类型的数据，其中也可以填入None，如果填入None则表示此位置可能是任何正整数。数据的batch大小不应包含在其中。

2. 传递一个batch_input_shape的关键字参数给第一层，该参数包含数据的batch大小。该参数在指定固定大小batch时比较有用，例如在stateful RNNs中。事实上，Keras在内部会通过添加一个None将input_shape转化为batch_input_shape

3. 有些2D层，如Dense，支持通过指定其输入维度input_dim来隐含的指定输入数据shape。一些3D的时域层支持通过参数input_dim和input_length来指定输入shape

```
model = Sequential()
model.add(Dense(32, input_shape=(784,)))

model = Sequential()
model.add(Dense(32, batch_input_shape=(None, 784)))
# note that batch dimension is "None" here,
# so the model will be able to process batches of any size.</pre>

model = Sequential()
model.add(Dense(32, input_dim=784))
```

```
model = Sequential()
model.add(LSTM(32, input_shape=(10, 64)))

model = Sequential()
model.add(LSTM(32, batch_input_shape=(None, 10, 64)))

model = Sequential()
model.add(LSTM(32, input_length=10, input_dim=64))
```
