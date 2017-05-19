import numpy as np
from keras.datasets import mnist
from keras.models  import Model
from keras.layers  import Input, Dense
from keras.optimizers import Adadelta
from keras.utils   import np_utils
from keras.utils.visualize_util import model_to_dot
from keras.utils.visualize_util import plot
from IPython.display import SVG
from keras import backend as K
from keras.callbacks import EarlyStopping


from matplotlib import pyplot as plt
plt.style.use("ggplot")

# Description
def draw_digit(data, row, col, n):
    size = int(np.sqrt(data.shape[0]))
    plt.subplot(row, col, n)
    plt.imshow(data.reshape(size, size))
    # plt.gray()

epochs = 20
batch_size = 258
input_dim = 28
input_unit_size = input_dim ** 2
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(X_train.shape[0], input_unit_size)
X_train = X_train.astype('float32') / 255

# Model definition
inputs = Input(shape=(input_unit_size,))
x = Dense(144, activation='relu')(inputs)
outputs = Dense(input_unit_size)(x)
model = Model(input=inputs, output=outputs)
model.compile(loss='binary_crossentropy', optimizer='adamax')

# training
early_stopping = EarlyStopping(monitor='loss', patience=5)
model.fit(X_train, X_train, callbacks=[early_stopping],
          nb_epoch=epochs, batch_size=batch_size)

# Draw original input data
show_size = 10
# total = 0
# plt.figure(figsize=(20, 20))
# for i in range(show_size):
#     for j in range(show_size):
#         draw_digit(X_train[total], show_size, show_size, total+1)
#         total += 1
# plt.show()

# Draw learning situation of hidden layer
get_layer_output = K.function([model.layers[0].input],
                              [model.layers[1].output])
hidden_outputs = get_layer_output([X_train[0:show_size**2]])[0]

total = 0
plt.figure(figsize=(20, 20))
for i in range(show_size):
    for j in range(show_size):
        draw_digit(hidden_outputs[total], show_size, show_size, total+1)
        total+=1
plt.show()

# Draw output of decoded output layer
# get_layer_output = K.function([model.layers[0].input],
#                               [model.layers[2].output])
# last_outputs = get_layer_output([X_train[0:show_size**2]])[0]
#
# total = 0
# plt.figure(figsize=(20, 20))
# for i in range(show_size):
#     for j in range(show_size):
#         draw_digit(last_outputs[total], show_size, show_size, total+1)
#         total+=1
# plt.show()


plot(model, to_file="model.png", show_shapes=True)