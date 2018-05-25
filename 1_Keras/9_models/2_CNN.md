```python
import numpy as np

np.random.seed(1337)  # for reproducibility

import keras
import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from keras.utils.vis_utils import plot_model
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adadelta


class KerasCNN(object):
    def __init__(self, X, y, optimizer=Adadelta, batch_size=128, nb_epoch=10):
        self.optimizer = optimizer
        self.input_shape = X.shape[1:]
        self.out_dim = y.shape[1]

        print(f"Input Dim: {self.input_shape}")
        print(f"Out Dim: {self.out_dim}\n")

        self.__build_keras_model()
        __callbacks = [self.checkpointer, self.lr_reducing, self.early_stopping]
        self.model.fit(X, y, batch_size, nb_epoch, verbose=1, callbacks=__callbacks, validation_split=0.2)

    def __build_keras_model(self):
        self.model = Sequential()
        self.model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=self.input_shape))
        self.model.add(Conv2D(64, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))
        self.model.add(Flatten())
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(self.out_dim, activation='softmax'))

        # self.model.load_weights(self.best_model_weight, by_name=True)
        self.model.compile(self.optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
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
        annealer = LearningRateScheduler(lambda x: 0.01 * 0.9 ** x)
        # annealer = ReduceLROnPlateau(factor=0.1, patience=10, verbose=1) # lr = lr*0.9
        return annealer

    @property
    def early_stopping(self):
        return EarlyStopping(patience=2, verbose=1)

#####################################################
f = np.load('/algor/yuanjie/mnist.npz')
X_train, y_train = f['x_train'], f['y_train']
X_test, y_test = f['x_test'], f['y_test']
X_train = X_train.reshape(-1, 28, 28, 1)/255
X_test = X_test.reshape(-1, 28, 28, 1)/255
# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)
#####################################################

KerasCNN(X_train, y_train)
```
