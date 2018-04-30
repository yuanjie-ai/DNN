class NLTKTool(object):

    def __init__(self, corpus):
        import nltk
        global nltk

        self.corpus = corpus


    def get_new_words(self, n=10, ngram='Bigram'):
        """
        :param n: 
        :param ngram: Bigram/Trigram
        :return: 
        """
        _getattr = lambda x: nltk.collocations.__getattribute__(x)
        _gram = _getattr('%sCollocationFinder' % ngram).from_words(self.corpus)
        return _gram.nbest(_getattr('%sAssocMeasures' % ngram).likelihood_ratio, n)
