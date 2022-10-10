# 第二章-PyTorch基础知识

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
  * 计算导数：在 Tensor 上调用 .backward()
* 梯度
  * TODO  

#### 小问题
1. 第 2.1 节，估计很多人是第一次看到“计算图”这个概念，我觉得可以考虑提供个链接供拓展阅读，或者说明后面会介绍到
2. 代码跟输出，一眼看去不好区分

