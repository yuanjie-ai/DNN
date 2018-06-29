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
```

