```python
inputImg = Input(shape=(784,))
encodingDim = 32
encodedIP = Dense(encodingDim, activation='relu')(inputImg)
decodedIP = Dense(784, activation='sigmoid')(encodedIP)

# encoder
encoder = Model(input=inputImg, output=encodedIP)

# autoencoder
autoencoder = Model(input=inputImg, output=decodedIP)

# decoder
encodedShaped = Input(shape=(encodingDim,))
decoder_layer = autoencoder.layers[-1]
decoder = Model(input=encodedShaped, output=decoder_layer(encodedShaped))

autoencoder.fit(X, X, nb_epoch=2, batch_size=256, validation_split=0.2)

encodedImgs = encoder.predict(X_test)
decodedImgs = decoder.predict(encodedImgs)

```
