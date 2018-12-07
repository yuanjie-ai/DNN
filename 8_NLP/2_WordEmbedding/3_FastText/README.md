
```python
from fastText.FastText import train_supervised
def get_test(path='./author_word_test.txt.txt'):
    df = pd.read_csv(path, '\t', names=['label', 'text'])
    X = df.text.values.tolist()
    y = df.label.apply(lambda x: int(x[-1])).values
    return X, y

model = train_supervised(
    './author_word_train.txt',
    lr=0.1,
    dim=128,
    ws=5,
    epoch=5,
    minCount=1,
    minCountLabel=0,
    minn=0,
    maxn=0,
    neg=5,
    wordNgrams=3,
    loss='softmax',
    bucket=2000000,
    thread=7,
    lrUpdateRate=100,
    t=0.0001,
    label='__label__',
    verbose=2,
    pretrainedVectors='../WordVectors/title_words.vector'
)

X, y = get_test('./author_word_test.txt')
preds = model.predict(X, 2)[1][:, 1]
auc(y, preds)
```
