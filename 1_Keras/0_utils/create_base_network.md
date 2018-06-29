```python
def create_base_network(input_shape):
    '''Base network to be shared (eq. to feature extraction).
    because we may re-use the same instance `base_network`, 
    the weights of the network will be shared across the many branches
    '''
    input = Input(shape=input_shape)
    x = Flatten()(input)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(128, activation='relu')(x)
    return Model(input, x)
    
```

