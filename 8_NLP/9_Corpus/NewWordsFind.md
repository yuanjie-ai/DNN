```python
# -*- coding: utf-8 -*-
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm


class NewWordsFind(object):
    def __init__(self, corpus, n=4, min_count=128, min_proba={2: 5, 3: 25, 4: 125}):
        self.n = n # 需要考虑的最长片段的字数（ngrams）
        self.min_count = min_count
        self.pattern_sub = re.compile('[^\u4e00-\u9fa5]+')  # 去除短文本
        self.pattern_split = re.compile('[^\u4e00-\u9fa5\w]+')  # 断句
        self.corpus = list(self.__corpus_convert(corpus))  # 爆内存？
        self.__ngrams()  # ngrams, total
        self.ngrams_ = set(i for i, j in self.ngrams.items() if self.__is_keep(i, min_proba))

    @property
    def words(self):
        words = defaultdict(int)
        for s in tqdm(self.corpus, desc='Cut Processing'):
            for i in self.__cut(s):
                words[i] += 1
        words = {i: j for i, j in words.items() if j >= self.min_count}
        return {i: j for i, j in words.items() if self.__is_real(i)}

    def __is_real(self, s):
        if len(s) >= 3:
            for i in range(3, self.n + 1):
                for j in range(len(s) - i + 1):
                    if s[j:j + i] not in self.ngrams_:
                        return 0
            return 1
        else:
            return 1

    def __cut(self, s):
        r = np.array([0] * (len(s) - 1))
        for i in range(len(s) - 1):
            for j in range(2, self.n + 1):
                if s[i: i + j] in self.ngrams_:
                    r[i: i + j - 1] += 1

        w = [s[0]]
        for i in range(1, len(s)):
            if r[i - 1] > 0:
                w[-1] += s[i]
            else:
                w.append(s[i])
        return w

    def __is_keep(self, s, min_proba):
        if len(s) >= 2:
            score = min([self.total * self.ngrams[s] / (self.ngrams[s[:i + 1]] * self.ngrams[s[i + 1:]]) for i in
                         range(len(s) - 1)])
            if score > min_proba[len(s)]:
                return 1
        else:
            return 0

    def __ngrams(self):
        ngrams = defaultdict(int)
        for s in tqdm(self.corpus, desc='Ngrams Processing'):
            l = len(s)
            for i in range(l):
                for j in range(1, self.n + 1):
                    if i + j <= l:
                        ngrams[s[i:i + j]] += 1

        self.ngrams = {i: j for i, j in ngrams.items() if j >= self.min_count}
        self.total = sum([j for i, j in ngrams.items() if len(i) == 1])

    def __corpus_convert(self, corpus):
        if isinstance(corpus, str) and Path(corpus).is_file():
            with open(corpus) as f:
                for doc in tqdm(f, desc="Corpus Convert Processing"):
                    assert isinstance(doc, str)
                    if len(self.pattern_sub.sub('', doc)) > 2:
                        for s in self.pattern_split.split(doc):
                            if s:
                                yield s
        else:
            for doc in tqdm(corpus, desc="Corpus Convert Processing"):
                assert isinstance(doc, str)
                if len(self.pattern_sub.sub('', doc)) > 2:
                    for s in self.pattern_split.split(doc):
                        if s:
                            yield s

```
