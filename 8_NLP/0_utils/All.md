```python
def file_processing(file_input, file_output='./tmp.txt', func=lambda x: x.strip()):
    from tqdm import tqdm
    with open(file_input) as ifile:
        for i in tqdm(ifile, desc='file_input'):
            with open(file_output, 'a') as ofile: # append
                ofile.writelines('%s\n' % func(i))
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
