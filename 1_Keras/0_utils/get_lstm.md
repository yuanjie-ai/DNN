```python
def get_lstm(input_shape=(3, 1)):
    '''
    data = array([0.1, 0.2, 0.3]).reshape((1,3,1))
    get_lstm().predict(data)
    '''
    input = Input(shape=input_shape)
    lstm = LSTM(1, return_sequences=True, return_state=True)
    return Model(input, lstm(input))
```
