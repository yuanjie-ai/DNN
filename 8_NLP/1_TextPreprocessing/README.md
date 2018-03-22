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
