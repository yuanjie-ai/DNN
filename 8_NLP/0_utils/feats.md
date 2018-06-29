## 特征工程
---
用leaning to rank做广告重点词


1、 词的tfidf 
2、 词在用户历史点击session中被替换的次数（subtfidf） 
3、 词性 
4、 前后词性（这个其实步长可以多几位，比如前前的词性，前前前的词性，实践中每多增加一种指标都会有提高，还是挺好用的） 
5、 [词的熵值]（一个term单独出现的频次越高，而且和其他term搭配出现的机会越少，那么我们可以肯定，这个term表达意图的能力越强，越重要）(https://www.zhihu.com/question/21104071/answer/36529209) 
熵怎么使用可以参看： 
http://blog.csdn.net/qjzcy/article/details/51728581 
6、crf训练的词的重要度（主要考虑是crf能很好的考虑前后词之间的关系） 
http://qtanalyzer.codeplex.com/ 
7、 是否停用词 
8、 是否数字 
9、 词在句子中的位置 
10、 词长 
11、 词是否前后缀词





----
- question(leak): 
  - tf: q1/q2/q1+q2
  - tfidf: q1/q2/q1+q2
  
- words(chars): 针对字符串计算
  - 词数
  - 词数差
  - 重叠词数：`len(set(q1) & set(q2))`
  - 相同度（相异度 = 1 - 相同度）: com / (q1 + q2 - com)每个状态分量根据目标设置最优权重
  - simhash
  - jaccard: `jaccard = lambda a, b: len(set(a).intersection(b))/(len(set(a).union(b))+0.)`
  - 对目标影响大的词（lstm状态差等）
  - 编辑距离
    - fuzz.QRatio
    - fuzz.WRatio
    - fuzz.partial_ratio
    - fuzz.token_set_ratio
    - fuzz.token_sort_ratio
    - fuzz.partial_token_set_ratio
    - fuzz.partial_token_sort_ratio
    - ...

  
- [doc2num][3]: 针对tf/tfidf/wordVectors等计算
  - n-grams: 结合tf/tfidf使用
  - gensim
    - wmd
    - norm_wmd(l2): norm_model.init_sims(replace=True)
  - skew/kurtosis: `from scipy.stats import skew, kurtosis`
  - scipy.spatial.distance: `import braycurtis, canberra, cityblock, cosine, euclidean, jaccard, minkowski`
  - cosine（修正）

- lda
- bleu（机器翻译指标）：对两个句子的共现词频率计算`torchtext`



---
[1]: https://ai.ppdai.com/mirror/goToMirrorDetail?mirrorId=1
[2]: https://github.com/Jie-Yuan/PpdaiQuestionPairsMatching/tree/master/Baseline
[3]: https://www.kaggle.com/kardopaska/fast-how-to-abhishek-s-features-w-o-cray-xk7
