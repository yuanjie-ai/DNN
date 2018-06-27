## 1. Keras范数
```python
# @KerasEval()
def l1(x):
    return K.sum(K.abs(x), axis=-1, keepdims=True)
    
# @KerasEval()
def l2(x):
    return K.sqrt(K.sum(K.square(x), axis=-1, keepdims=True))
```

## 1. 余弦距离
```python
def np_cos(a, b, scale=False):
    cos = np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))
    if scale:
         cos = 0.5*(cos + 1)
    return cos
    
# @KerasEval()
def keras_cos(a, b):
    return K.dot(a, K.transpose(b))/(l2(a)*l2(b))
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
