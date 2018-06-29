```python
def create_shared_layer(input_shape):
    """Base network to be shared (eq. to feature extraction).
    because we may re-use the same instance `base_network`, 
    the weights of the network will be shared across the many branches
    
    :param input_shape: X.shape[1:]
    """
    input = Input(shape=input_shape)
    ####################################
    x = Flatten()(input)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.1)(x)
    ####################################
    return Model(input, x)

input_a, input_b = Input(shape=input_shape), Input(shape=input_shape)
shared_layer = create_shared_layer()
output_a, output_b = shared_layer(input_a), shared_layer(input_b)

# We can then concatenate the two vectors:
merged_vector = keras.layers.concatenate([output_a, output_b], axis=-1)

# And add a logistic regression on top
predictions = Dense(1, activation='sigmoid')(merged_vector)

# We define a trainable model linking the
# tweet inputs to the predictions
model = Model(inputs=[tweet_a, tweet_b], outputs=predictions)

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.fit([data_a, data_b], labels, epochs=10)
```

