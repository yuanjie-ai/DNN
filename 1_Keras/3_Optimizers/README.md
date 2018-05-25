<h1 align = "center">:rocket: Optimizers :facepunch:</h1>

---
## `keras.optimizers`
- Adadelta
- Adagrad
- Adam
- Adamax
- Nadam
- RMSprop
- SGD
- TFOptimizer

---
## 动态学习率
```python
import keras.backend as K
from keras.callbacks import LearningRateScheduler
annealer = LearningRateScheduler(lambda x: K.get_value(self.model.optimizer.lr) * 0.9 ** x)
self.model.fit(X, y, batch_size, nb_epoch, verbose=2, callbacks=[ annealer], validation_split=0.2)
```

