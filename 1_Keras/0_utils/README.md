<h1 align = "center">:rocket: KERAS API :facepunch:</h1>

---
## `import keras.preprocessing.text as T`
- `T.maketrans`
```python
'aabbcc'.translate(T.maketrans('abc', '123'))
'cba'.translate(str.maketrans("abc", "123"))
```
- `T.one_hot`
- `T.text_to_word_sequence`
- `T.Tokenizer`
  ```python
  text1='some thing to eat'
  text2='some thing to drink'
  texts=[text1,text2]
  tokenizer = T.Tokenizer(10**6)
  tokenizer.fit_on_texts(texts)
  
  tokenizer.word_counts
  tokenizer.word_docs
  tokenizer.word_index
  tokenizer.index_docs
  
  tokenizer.texts_to_matrix(texts, mode='binary') # mode: one of "binary", "count", "tfidf", "freq".
  (tokenizer.texts_to_sequences_generator(texts)
  ```

---
