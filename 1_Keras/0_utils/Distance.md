## Keras范数
```python
# @KerasEval()
def l1(x):
    return K.sum(K.abs(x), axis=-1, keepdims=True)
    
# @KerasEval()
def l2(x):
    return K.sqrt(K.sum(K.square(x), axis=-1, keepdims=True))
```

---
## 1. 余弦距离
```python
# @KerasEval()
def keras_cos(a, b, scale=True):
    dist = K.dot(a, K.transpose(b))/(l2(a)*l2(b))
    if scale:
        dist = 0.5*(K.sum(cos, axis=-1, keepdims=True) + 1)
    return dist
```
## 2. 欧式距离
```python
```

## 3. 曼哈顿距离
```python
# @KerasEval()
def manhattan(left, right, scale=True):
    dist = K.sum(K.abs(left - right), keepdims=True)
    if scale:
        dist = K.exp(-dist)
    return dist
```

---
```python
class KerasEval:
    def __init__(self):
        print('Get Keras Eval Func!!!')

    def __call__(self, func):
        self.func = func

        def realfunc(*args):
            return K.eval(self.func(*map(K.variable, args)))

        return realfunc
```
