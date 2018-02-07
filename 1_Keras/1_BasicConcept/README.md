<h1 align = "center">:rocket: 基本概念 :facepunch:</h1>

---
## 1. Tensor
- 定义：张量可以看作是向量、矩阵的自然推广，我们用张量来表示广泛的数据类型
- 阶数：维度/轴/axis

  > 譬如一个矩阵[[1,2],[3,4]]，是一个2阶张量，有两个维度或轴，沿着第0个轴（为了与python的计数方式一致，本文档维度和轴从0算起）你看到的是[1,2]，[3,4]两个向量，沿着第1个轴你看到的是[1,3]，[2,4]两个向量。

---
## 2. Data_format
- TensorFlow: 数据组织方式称为channels_last(~/.keras/keras.json)
  > 表达形式是（100,16,32,3）：（样本维，高，宽，通道维）
  
---
## 3. Model
- SequentialModel: Sequential其实是Graph的一个特殊情况
- FunctionalModel(Graph): 这个模型支持多输入多输出，层与层之间想怎么连怎么连，但是编译速度慢

---
## 4. Batch
- Batch gradient descent: 批梯度下降
- Stochastic gradient descent: 随机梯度下降
- Mini-batch gradient decent: 小批的梯度下降

---
## 5. Epochs: 训练过程中数据将被“轮”多少次
