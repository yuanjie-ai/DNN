```python
import keras.backend as K

class KerasEval:
    def __init__(self):
        print('Get Keras Eval Func!!!')

    def __call__(self, func):
        self.func = func

        def realfunc(*args, **kwargs):
            """
            args: tensor
            kwargs: not tensor
            """
            return K.eval(self.func(*map(K.variable, args), **kwargs))

        return realfunc
        
@KerasEval()
def euclidean(a, b, scale=True):
    dist = K.sqrt(K.sum(K.square(a - b), keepdims=True))
    if scale:
        dist = 1/(1 + dist)
    return dist
    
euclidean([[1,2]], [[1,4]], scale=False)
euclidean([[1,2]], [[1,4]], scale=True)
```
