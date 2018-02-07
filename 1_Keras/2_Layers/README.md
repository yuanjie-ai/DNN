<h1 align = "center">:rocket: 网络层 :facepunch:</h1>

---
http://blog.csdn.net/xiaozhuge080/article/details/52678453
http://blog.csdn.net/pengjian444/article/details/56316445
---
## 1. 常用层：`keras.layers.core`

- Dense层：全连接层
- Activation层：对一个层的输出添加激活函数
- Dropout层：每次更新参数的时候随机断开一定百分比(b)的输入神经元连接，用于防止过拟合
- Flatten层：用来将输入“压平”，即把多维的输入一维化，常用在从卷积层到全连接层的过渡。
- Reshape层：用来将输入shape转换为特定的shape
- Permute层：将输入的维度按照给定模式进行重排，例如，当需要将RNN和CNN网络连接时，可能会用到该层
- RepeatVector层：RepeatVector层将输入重复n次
- Merge层：Merge层根据给定的模式，将一个张量列表中的若干张量合并为一个单独的张量
- Lambda层：本函数用以对上一层的输出施以任何Theano/TensorFlow表达式
- ActivityRegularizer层：经过本层的数据不会有任何变化，但会基于其激活值更新损失函数值
- Masking层：使用给定的值对输入的序列信号进行“屏蔽”，用以定位需要跳过的时间步。
- Highway层：Highway层建立全连接的Highway网络，这是LSTM在前馈神经网络中的推广
- MaxoutDense层：参数尚不理解，具体参考文献和文档。
