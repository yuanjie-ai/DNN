```python
from keras.callbacks import *


class KerasCallbacks(object):
    def model_checkpoint(self, best_model_weight="best_model_weights.hdf5"):
        return ModelCheckpoint(best_model_weight, verbose=0, save_best_only=True)

    def lr_scheduler(self):
        annealer = LearningRateScheduler(lambda x: 0.01 * 0.9 ** x)  # Dynamic learning rate
        # keras.callbacks.ReduceLROnPlateau
        return annealer

    def early_stopping(self):
        EarlyStopping()
        pass

```
