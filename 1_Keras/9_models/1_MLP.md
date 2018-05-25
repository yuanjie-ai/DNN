```python
import numpy as np

np.random.seed(1337)  # for reproducibility

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.utils.vis_utils import plot_model
from keras.callbacks import ModelCheckpoint


class KerasMLP(object):
    def __init__(self, X, y, batch_size=32, nb_epoch=10, verbose=1, best_model_weight="best_model_weights.hdf5"):
        self.input_dim = X.shape[1]
        self.out_dim = y.shape[1]
        self.best_model_weight = best_model_weight
        print(f"Input Dim: {self.input_dim}")
        self.__build_keras_model()
        # filepath="weights-improvement-{epoch}-{val_acc:.2f}.hdf5" # 多个check point
        self.checkpointer = ModelCheckpoint(self.best_model_weight, verbose=0, save_best_only=True)
        self.model.fit(X, y, batch_size, nb_epoch, verbose, callbacks=[self.checkpointer], validation_split=0.2)

    def __build_keras_model(self):
        self.model = Sequential()
        self.model.add(Dense(64, init="uniform", activation='relu', name="dense_1", input_dim=self.input_dim))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(32, init="uniform", activation='relu', name="dense_2"))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(self.out_dim, activation='sigmoid', name="dense_3"))
        
        # self.model.load_weights(self.best_model_weight, by_name=True)
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        self.model.summary()

    def plot_model(self):
        plot_model(self.model, to_file='model.png')

```
