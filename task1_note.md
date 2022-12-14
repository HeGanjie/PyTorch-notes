# Task1-PyTorch基础知识（第二章）

[教程原文](https://datawhalechina.github.io/thorough-pytorch/%E7%AC%AC%E4%BA%8C%E7%AB%A0/index.html)

#### 关键要点

* 张量
  * 类似 numpy 的多维数组，但是提供GPU计算和自动求梯度等更多功能，使 Tensor 这一数据类型更加适合深度学习
  * 基础用法: `import torch` 然后 `torch.rand`
  * 常用函数：rand(4,3)、zeros(4,3)、tensor([5.5, 3])、ones(4,3)、randn_like(tensor)、arange(s,e,step)、randperm(m)
  * 运算：t1 + t2、t1.add(t2)
  * 副作用运算：t1.add_(t2)
  * 需要注意：索引出来的结果与原数据共享内存，如果不想修改可以使用copy()
  * 维度变换：view()，实际内存还是共享的，如果不想共享，先 clone() 再 view()
  * clone：会被记录在计算图中，即梯度回传到副本时也会传到源 Tensor
  * 取值：t1.item()
  * 广播机制：运算时会自动根据最终的 shape，先对变量进行重复操作
* 自动求导
  * 如果 Tensor 的属性 .requires_grad 为 True，那么它将会追踪对于该张量的所有操作。当完成计算后可以通过调用 .backward()，来自动计算所有的梯度。这个张量的所有梯度将会自动累加到.grad属性
  * 要阻止一个张量被跟踪历史，可以调用.detach()方法将其与计算历史分离，并阻止它未来的计算记录被跟踪
  * 为了防止跟踪历史记录(和使用内存），可以将代码块包装在 with torch.no_grad(): 中
  * 无环图（计算图？）：每个张量都有一个.grad_fn属性，该属性引用了创建 Tensor 自身的Function(除非这个张量是用户手动创建的，即 grad_fn是 None )，Function 再连接之前的 Tensor
  * 计算导数：在 Tensor 上调用 .backward()，非标量的话需要传入 gradient 参数，该参数是形状匹配的张量
  * 梯度，注意：grad在反向传播过程中是累加的(accumulated)，这意味着每一次运行反向传播，梯度都会累加之前的梯度，所以一般在反向传播之前需把梯度清零 `x.grad.data.zero_()`
  * 如果想修改 tensor 的值，但是又不希望被 autograd 记录(即不会影响反向传播)， 那么可以对 tensor.data 进行操作
* 并行计算
  * CUDA是NVIDIA提供的GPU并行计算框架，为啥用起来是 `.cuda()` 而不是 `.gpu()` 是因为目前只支持这个
  * 当我们使用了 .cuda() 时，其功能是让我们的模型或者数据从CPU迁移到GPU(0)当中，通过GPU开始计算
  * 数据在GPU和CPU之间进行传递时会比较耗时，我们应当尽量避免数据的切换
  * GPU运算很快，但是在使用简单的操作时，我们应该尽量使用CPU去完成
  * 当我们的服务器上有多个GPU，我们应该指明我们使用的GPU是哪一块
  * 主流方式是数据并行的方式(Data parallelism)：不同的数据分布到不同的设备中，执行相同的任务

#### 教程的小建议
1. 第 2.1 节，估计很多人是第一次看到“计算图”这个概念，我觉得可以考虑提供个链接供拓展阅读，或者说明后面会介绍到
2. 代码跟输出，一眼看去不好区分，建议参考 pandas 文档，在代码前加 >>>
3. 雅可比向量积的例子有点难懂，忘了 norm 是什么意思了，查了才知道是二范数；或者考虑举一些具体些的公式例子，例如自由落体高度公式
4. 建议变量的引用范围不要太远，2.2.1 一开始的 out 变量的值是什么，我还得往前翻找一阵才知道
5. backward 如果带参数，没有讲拿到雅可比向量积后可以做什么，有点迷
