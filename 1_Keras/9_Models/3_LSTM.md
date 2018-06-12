```python
import numpy as np

np.random.seed(1337)  # for reproducibility

import keras
import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM
from keras.utils.vis_utils import plot_model
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adadelta, Adam


class KerasLSTM(object):
    def __init__(self, X, y, max_feature=10000, optimizer=Adadelta(2), batch_size=128, nb_epoch=10):
        self.max_feature = max_feature
        self.optimizer = optimizer
        self.input_shape = X.shape[1:]
        self.out_dim = y.shape[1]

        print(f"Input Dim: {self.input_shape}")
        print(f"  Out Dim: {self.out_dim}\n")

        self.__build_keras_model()
        __callbacks = [self.checkpointer, self.lr_reducing, self.early_stopping]
        self.model.fit(X, y, batch_size, nb_epoch, verbose=1, callbacks=__callbacks, validation_split=0.25)

    def __build_keras_model(self):
        self.model = Sequential()
        self.model.add(Embedding(self.max_feature, 128, mask_zero=True))
        self.model.add(LSTM(128))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(self.out_dim, activation='sigmoid')) # 'softmax'需要onehot

        # self.model.load_weights(self.best_model_weight, by_name=True)
        self.model.compile(self.optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        self.model.summary()

    def plot_model(self):
        plot_model(self.model, to_file='model.png')

    """class KerasCallbacks(object):"""
    @property
    def checkpointer(self):
        # filepath="weights-improvement-{epoch}-{val_acc:.2f}.hdf5" # 多个check point
        return ModelCheckpoint("best_model_weights.hdf5", verbose=1, save_best_only=True)

    @property
    def lr_reducing(self):
        """Dynamic learning rate
        """
        annealer = LearningRateScheduler(lambda x: 1 * 0.9 ** x)
#         annealer = ReduceLROnPlateau(factor=0.1, patience=3, verbose=1) # lr = lr*0.9
        return annealer

    @property
    def early_stopping(self):
        return EarlyStopping(patience=3, verbose=1)

```
