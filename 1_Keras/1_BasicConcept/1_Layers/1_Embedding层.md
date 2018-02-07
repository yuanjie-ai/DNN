[<h1 align = "center">:rocket: Embedding层 :facepunch:</h1>][1]

---
> Embedding层：以OneHot为输入，中间层节点为字/词向量维数的全连接层，而这个全连接层的参数，就是一个“字/词向量表”！

- 运算层面：因为把OneHot型的矩阵运算简化为了查表操作，降低了运算量不是因为词向量的出现
- 逻辑层面：它得到了这个全连接层的参数之后，直接用这个全连接层的参数作为特征，或者说，用这个全连接层的参数作为字、词的表示，从而得到了字、词向量，最后还发现了一些有趣的性质，比如向量的夹角余弦能够在某种程度上表示字、词的相似度。

---
## [Keras][2]
```python
from keras.layers.embeddings import Embedding
```





---
[1]: https://spaces.ac.cn/archives/4122/
[2]: http://keras-cn.readthedocs.io/en/latest/layers/embedding_layer/


