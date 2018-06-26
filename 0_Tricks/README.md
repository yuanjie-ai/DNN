## 1. Batchsize
> 批量大小将决定我们一次训练的样本数。

- Mini-batch（小批次）：batch_size的正确选择是为了在内存效率和内存容量之间寻找最佳平衡。
- Batch（全批次）：https://blog.csdn.net/s_sunnyy/article/details/65445197
- Stochastic（随机batch_size=1）：每次修正方向以各自样本的梯度方向修正，振荡难以收敛。
- Epoch: 所有训练样本的一个正向传递和一个反向传递。
- Iteration: 每一次迭代得到的结果都会被作为下一次迭代的初始值。一个迭代 = 一个正向通过+一个反向通过

- 适当的增加BatchSize的优点：
    - 相对于正常数据集，如果Batch_Size过小，训练数据就会非常难收敛，从而导致underfitting。
    - 单次epoch的迭代次数减少，提高运行速度。（单次epoch=（全部训练样本/batchsize） / iteration = 1）
    - 梯度下降方向准确度增加，训练震动的幅度减小。
    - 通过并行化提高内存利用率，相对处理速度加快。
    - 所需内存容量增加（epoch的次数需要增加以达到最好结果）。
    - 这里我们发现上面两个矛盾的问题，因为当epoch增加以后同样也会导致耗时增加从而速度下降。因此我们需要寻找最好的batch_size。

---

