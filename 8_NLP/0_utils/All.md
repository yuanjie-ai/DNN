```python
def file_processing(file_input, file_output='./tmp.txt', func=lambda x: x.lower()):
    from tqdm import tqdm
    with open(file_input) as ifile:
        with open(file_output, 'a') as ofile: # append
            for i in tqdm(ifile, desc='File Processing'):
                if i.strip():
                    ofile.writelines(func(i))
```
---
```python
def get_tfidf(sentence, tfidf_model):
    """
    :param sentence: ['w1', 'w2']
    :param tfidf_model:
    :return:
    """
    _dict = tfidf_model.vocabulary_
    M = tfidf_model.transform([' '.join(sentence)])
    return sorted([(M[:, _dict[i]].max(), i) for i in sentence if i in _dict])[::-1]
```
