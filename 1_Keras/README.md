LSTM情感分析实验复现（一）one-hot encoding
http://blog.csdn.net/xhyqlbd/article/details/79006899



```python
# http://willwolf.io/2017/05/08/transfer-learning-flight-delay-prediction/
from abc import ABCMeta, abstractmethod

from keras import Input, Model
from keras.layers import Dense, Dropout
from keras.regularizers import l2
```
