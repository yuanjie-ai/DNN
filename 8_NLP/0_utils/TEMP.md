# stem.LancasterStemmer
# stem.PorterStemmer
# stem.SnowballStemmer

# stem.RegexpStemmer
# stem.WordNetLemmatizer # 推荐
s = 'I went homes! taking apples'
words = nltk.tokenize.word_tokenize(s)
words_pos = nltk.pos_tag(words)
words_pos = [(w, 'v') if p[0]=='V' else (w, 'n') for w, p in words_pos]
# stem词干化
[stem.WordNetLemmatizer().lemmatize(*i) for i in words_pos]
