## 特征工程
- 根据Tfidf/词频OneHot
- total_length
- num_words
- num_unique_words
- words_vs_unique: num_unique_words/num_words
- num_exclamation_marks: 感叹号
- num_question_marks: 问号
- num_punctuation: 标点
- num_symbols
- num_smilies

---
## 文本清洗
### 1. 正则
```python
pattern_char = re.compile('[A-Za-z]+')
pattern_chinese = re.compile('[\u4e00-\u9fa5]+')
# pattern = re.compile('[^A-Za-z\u4e00-\u9fa5]+')
```

### 2. 清洗
```python
import re
import jieba_fast as jieba
from pipe import *

def read(file):
    with open(file) as f:
        return f.read()

def write(text, file, overwrite=True):
    if overwrite:
        with open(file, 'w') as f:
            f.write(text)
    else:
        with open(file, 'a') as f:
            f.write(text)

pattern = re.compile('[^0-9A-Za-z\u4e00-\u9fa5]+')
cut = Pipe(lambda x: jieba.lcut(x))
sub = Pipe(lambda x: pattern.sub(' ', x))

text_clean = lambda file_path: read(file_path).replace('\n', '').lower() | sub | cut | concat(' ')

file_path = '/DATA/1_DataCache/NLP/Corpus/new_zhwiki/test.txt'
write(text_clean(file_path), '/DATA/1_DataCache/NLP/Corpus/new_zhwiki/wikiCleaned.txt')
```
