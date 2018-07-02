```python
def get_lstm():
'''
data = array([0.1, 0.2, 0.3]).reshape((1,3,1))
get_lstm().predict(data)
'''
    input = Input(shape=(3, 1))
    x = LSTM(1, return_sequences=True, return_state=True)(input)
    return Model(input, x)
```
