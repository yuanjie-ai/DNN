https://blog.csdn.net/u011327333/article/details/78501054

https://blog.csdn.net/jiangpeng59/article/details/77646186



`lstm1, state_h, state_c = LSTM(1, return_sequences=True, return_state=True)`
- `lstm1`: 存放的就是全部时间步的 hidden state
- `state_h`: 存放的是最后一个时间步的 hidden state
- `state_c`: 存放的是最后一个时间步的 cell state
