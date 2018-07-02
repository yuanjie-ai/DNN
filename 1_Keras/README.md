<h1 align = "center">:rocket: Keras :facepunch:</h1>

---
## Model
- [Save/Load][1]
- Predict
```python
model.predict_classes # np.argmax(model.preidct(X), axis=1)
```
## Layers
### [LSTM参数详解][4]
- `LSTM(units, dropout=0.0, recurrent_dropout=0.0, return_sequences=False, return_state=False, stateful=False)`
  - `return_sequences=True`
  
## [LSTM情感分析实验][2]

## [封装][3]


---
[1]: https://blog.csdn.net/jiandanjinxin/article/details/77152530
[2]: http://blog.csdn.net/xhyqlbd/article/details/79006899
[3]: http://willwolf.io/2017/05/08/transfer-learning-flight-delay-prediction/
[4]: https://blog.csdn.net/jiangpeng59/article/details/77646186
