<h1 align = "center">:rocket: Keras :facepunch:</h1>

---
## Model
- [Save/Load][1]
- Predict
```python
model.predict_classes # np.argmax(model.preidct(X), axis=1)
```
## Layers
### [LSTM参数详解][4]: 
- `LSTM(units, dropout=0.0, recurrent_dropout=0.0, return_sequences=False, return_state=False, stateful=False)`
  - `return_sequencese`: 在输出序列中，返回单个 hidden state值还是返回全部time step 的 hidden state值。 False 返回单个， true 返回全部。
  True返回形如`(samples, timesteps, output_dim)`的3D张量否则，返回形如`(samples, output_dim)`的2D张量，我们可以把很多   LSTM层串在一起，但是最后一个 LSTM层return_sequences通常为False
  - `return_state`: 除了输出之外是否返回最后一个状态
  
  
## [LSTM情感分析实验][2]

## [封装][3]


---
[1]: https://blog.csdn.net/jiandanjinxin/article/details/77152530
[2]: http://blog.csdn.net/xhyqlbd/article/details/79006899
[3]: http://willwolf.io/2017/05/08/transfer-learning-flight-delay-prediction/
[4]: https://blog.csdn.net/jiangpeng59/article/details/77646186
