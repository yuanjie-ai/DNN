```python
# Draw learning situation of hidden layer
get_layer_output = K.function(inputs, outputs)  # ([model.layers[0].input], [model.layers[1].output])
hidden_outputs = get_layer_output(inputs)[0]  # get_layer_output([X_train])
```
