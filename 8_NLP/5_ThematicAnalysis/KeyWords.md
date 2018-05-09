<h1 align = "center">:rocket: 关键词提取 :facepunch:</h1>

---
## 1. CV

---
## 2. TFIDF

---
## 3. TextRank

---
## [4. Glove][1]

---
## 5. Skip-Gram + Huffman Softmax
```python
from pathlib import Path

import gensim
import numpy as np


class SGHS(object):
    def __init__(self, sg_hs_model=None):
        """
        :param sg_hs_model: Skip-Gram + Huffman Softmax
        """
        self.model = sg_hs_model
        self._model_convert()

    def key_words(self, s):
        s = [i for i in s if self.model.wv.__contains__(i)]
        ws = [(i, sum([self._prob(o, i) for o in s])) for i in s]

        return sorted(ws, key=lambda x: x[1])[::-1]

    def _prob(self, oword, iword):
        x = self.model.wv.word_vec(iword)  # 输入词向量
        oword = self.model.wv.vocab[oword]
        d = oword.code  # 该节点的编码（非0即1）
        p = oword.point  # 该节点的Huffman编码路径

        theta = self.model.trainables.syn1[p].T  # size*n: 300*4

        dot = np.dot(x, theta)  # 4*4
        lprob = -sum(np.logaddexp(0, -dot) + d * dot)  # 估算词与词之间的转移概率就可以得到条件概率了

        return lprob

    def _model_convert(self):
        if isinstance(self.model, str) and Path(self.model).is_file():
            self.model = gensim.models.Word2Vec.load(self.model)

```

---
[1]: https://github.com/Jie-Yuan/AI/blob/master/8_NLP/2_WordEmbedding/2_Glove/load_glove.md
