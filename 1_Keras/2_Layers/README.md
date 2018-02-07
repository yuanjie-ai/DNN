<h1 align = "center">:rocket: 网络层 :facepunch:</h1>

---
## 常用层：`keras.layers.core`

### 1. Dense层：全连接层
### 2. Activation层：激活层对一个层的输出施加激活函数
### 3. Dropout层
> 为输入数据施加Dropout。Dropout将在训练过程中每次更新参数时随机断开一定百分比（p）的输入神经元连接，Dropout层用于防止过拟合。
### 4. Flatten层
> Flatten层用来将输入“压平”，即把多维的输入一维化，常用在从卷积层到全连接层的过渡。Flatten不影响batch的大小。
### 5.Reshape层: Reshape层用来将输入shape转换为特定的shape
