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

---
## 2. Convolution
- Convolution2D层：二维卷积层对二维输入进行滑动窗卷积
- AtrousConvolution2D层：该层对二维输入进行Atrous卷积，也即膨胀卷积或带孔洞的卷积。
    - Convolution1D, AtrousConvolution1D，Convolution3D同
- SeparableConvolution2D层：该层是对2D输入的可分离卷积。可分离卷积首先按深度方向进行卷积（对每个输入通道分别卷积），然后逐点进行卷积，将上一步的卷积结果混合到输出通道中。
- Deconvolution2D层：该层是卷积操作的转置（反卷积）。需要反卷积的情况通常发生在用户想要对一个普通卷积的结果做反方向的变换。例如，将具有该卷积层输出shape的tensor转换为具有该卷积层输入shape的tensor。
- Cropping1D层：在时间轴（axis1）上对1D输入（即时间序列）进行裁剪
- Cropping2D层：对2D输入（图像）进行裁剪，将在空域维度，即宽和高的方向上裁剪
- Cropping3D层：对2D输入（图像）进行裁剪
- UpSampling1/2/3D层：不明所以
- ZeroPadding1D层：对1D输入的首尾端（如时域序列）填充0，以控制卷积以后向量的长度
- ZeroPadding2D层：对2D输入（如图片）的边界填充0，以控制卷积以后特征图的大小
- ZeroPadding3D层：将数据的三个维度上填充0

---
## 3. Pooling
- MaxPooling1D层：对时域1D信号进行最大值池化
- MaxPooling2D层：为空域信号施加最大值池化
- MaxPooling3D层：为3D信号（空域或时空域）施加最大值池化
- AveragePooling1/2/3D层
- GlobalMaxPooling1/2D层
- GlobalAveragePooling1/2D层

---
## 4. LocallyConnceted
- LocallyConnected1/2D层：和 Convolution1/2D工作方式类似，唯一不同的是不进行权值共享。

## 5. Recurrent
- Recurrent层：这是递归层的抽象类，不能实例化，请使用它的子类：LSTM/SimpleRNN
- SimpleRNN层：全连接RNN网络，RNN的输出会被回馈到输入
- GRU层：门限递归单元（详见参考文献）
- LSTM层：Keras长短期记忆模型，关于此算法的详情，请参考[本教程][1]








---
[1]: http://deeplearning.net/tutorial/lstm.html










