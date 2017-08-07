# :rocket: TensorflowLearning :facepunch:
---

## TfDeploy
- Dump
```
import tensorflow as tf
import tfdeploy as td

# setup tfdeploy (only when creating models)
td.setup(tf)

# build your graph
sess = tf.Session()
# use names for input and output layers
x = tf.placeholder("float", shape=[None, 784], name="input")
W = tf.Variable(tf.truncated_normal([784, 100], stddev=0.05))
b = tf.Variable(tf.zeros([100]))
y = tf.nn.softmax(tf.matmul(x, W) + b, name="output")
sess.run(tf.global_variables_initializer())

model = td.Model()
model.add(y, sess) # y and all its ops and related tensors are added recursively
model.save("model.pkl")
```
- Load
```
import numpy as np
import tfdeploy as td

model = td.Model("model.pkl")
# shorthand to x and y
x, y = model.get("input", "output")
# evaluate
batch = np.random.rand(10000, 784)
result = y.eval({x: batch})
```
