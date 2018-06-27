```python
import keras.backend as K

class KerasEval:
    def __init__(self):
        print('Get Keras Eval Func!!!')
    
    def __call__(self, func):
        self.func = func
        def realfunc(*args):
            return K.eval(self.func(*map(K.variable, args)))

        return realfunc
```
