```python
from keras.callbacks import *


class KerasCallbacks(object):

    @property
    def checkpointer(self):
        # filepath="weights-improvement-{epoch}-{val_acc:.2f}.hdf5" # 多个check point
        return ModelCheckpoint("best_model_weights.hdf5", verbose=1, save_best_only=True)

    @property
    def lr_reducing(self):
        """Dynamic learning rate
        """
        annealer = LearningRateScheduler(lambda x: 0.01 * 0.9 ** x)
        # annealer = ReduceLROnPlateau(factor=0.1, patience=10, verbose=0) # lr = lr*0.9
        return annealer

    @property
    def early_stopping(self):
        return EarlyStopping(patience=2, verbose=1)
```
