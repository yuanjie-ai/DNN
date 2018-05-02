```python
def word_bag(corpus):
    """
    二维
    """
    from gensim.corpora import Dictionary
    dct = Dictionary(corpus)
#     dct.add_documents([["cat", "say", "meow"], ["dog"]])  # update dictionary with new documents
    dct.doc2bow(["dog", "computer", "non_existent_word"])

# dct.doc2bow
# dct.doc2idx
# dct.id2token
# dct.token2id
```
