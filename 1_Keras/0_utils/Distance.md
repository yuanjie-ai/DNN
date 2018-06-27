## 1. Keras范数
```python
@KerasEval()
def l1(x):
    return K.sum(K.abs(x), axis=-1, keepdims=True)
    
@KerasEval()
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
```
