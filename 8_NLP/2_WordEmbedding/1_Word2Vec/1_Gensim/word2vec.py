from gensim.models import Word2Vec

sentences = [('word', 'segment'), ('word', 'segment')]
SkipGram = Word2Vec(
    sentences,
    size=300,
    window=3,
    min_count=100,
    workers=20,
    sg=0,
    hs=0,
    negative=5,
    iter=10
)
CBOW = Word2Vec(
    sentences,
    size=300,
    window=3,
    min_count=100,
    workers=20,
    sg=1,
    hs=0,
    negative=5,
    iter=10
)
