# Task2-Pytorch的各个组件和实战（第三、四章）

#### 关键要点

* 机器学习开发流程  
  1. 数据预处理
  2. 选择模型，损失函数、优化方法以及超参数
  3. 拟合、验证
* 深度学习与机器学习的主要区别
  1. 数据量一般很大，所以需要分批进行训练  
  2. 模型层数更多，有实现特定功能的层，且往往需要“逐层”搭建；
  3. 损失函数和优化器要能够保证反向传播能够在用户自行定义的模型结构上实现
* 基本配置
  * 引入PyTorch常用模块：torch、torch.nn、torch.utils.data.Dataset、torch.utils.data.DataLoader、torch.optimizer
  * 引入其他常用库：os、numpy、pandas、matplotlib、seaborn、sklearn
  * 配置常用超参数：batch_size、lr、max_epochs 
  * 其他：配置启用 cuda
* 数据读入
  * 是通过Dataset+DataLoader的方式完成的，Dataset定义好数据的格式和数据变换形式，DataLoader用iterative的方式不断读入批次数据
  * 可以定义自己的Dataset类来实现灵活的数据读取，定义的类需要继承PyTorch自身的Dataset类
* 模型构建
  * PyTorch中神经网络构造一般是基于 Module 类的模型来完成的，它让模型构造更加灵活
  * 神经网络的构造
    * Module 类是 nn 模块里提供的一个模型构造类，是所有神经⽹网络模块的基类，我们可以继承它来定义我们想要的模型
    * Module 的子类既可以是⼀个层(如PyTorch提供的 Linear 类)，⼜可以是一个模型(如这里定义的 MLP 类)，或者是模型的⼀个部分
  * 神经网络中常见的层
    * 自定义层
      * 可以没有参数；
      * 若带参数，应定义成 Parameter，会⾃动被添加到模型的参数列表里
      * 还可以使⽤ ParameterList 和 ParameterDict 分别定义参数的列表和字典
    * 卷积层：二维卷积层将输入和卷积核做互相关运算，并加上一个标量偏差来得到输出。卷积层的模型参数包括了卷积核和标量偏差。在训练模型时，通常先对卷积核随机初始化，然后不断迭代卷积核和偏差
    * 池化层：池化层每次对输入数据的一个固定形状窗口(⼜称池化窗口)中的元素计算输出，例如最大值或平均值
    * 全连接层、与循环层
  * 模型示例
    * 神经网络的典型训练过程如下：
      1. 定义包含一些可学习参数(或者叫权重）的神经网络
      2. 在输入数据集上迭代
      3. 通过网络处理输入
      4. 计算 loss (输出和正确答案的距离）
      5. 将梯度反向传播给网络的参数
      6. 更新网络的权重，一般用：weight -= learning_rate * gradient
    * 只需要定义 forward 函数（可使用任何针对张量的操作和计算），backward函数会在使用autograd时自动定义（用来计算导数）
    * 一个模型的可学习参数可以通过net.parameters()返回
    * 注意：torch.nn只支持小批量处理 (mini-batches），不支持单个样本的输入。比如，nn.Conv2d 接受一个4维的张量，即nSamples x nChannels x Height x Width。如果是一个单独的样本，则需使用input.unsqueeze(0) 来添加一个“假的”批大小维度
    * 常用类：
      1. torch.Tensor - 一个多维数组，支持诸如backward()等的自动求导操作，同时也保存了张量的梯度
      2. nn.Module - 神经网络模块。是一种方便封装参数的方式，具有将参数移动到GPU、导出、加载等功能
      3. nn.Parameter - 张量的一种，当它作为一个属性分配给一个Module时，它会被自动注册为一个参数
      4. autograd.Function - 实现了自动求导前向和反向传播的定义，每个Tensor至少创建一个Function节点，该节点连接到创建Tensor的函数并对其历史进行编码
* 模型初始化    
  * 在深度学习模型的训练中，权重的初始值极为重要。一个好的权重值，会使模型收敛速度提高，使模型准确率更精确
  * 为了利于训练和减少收敛时间，我们需要对模型进行合理的初始化
  * PyTorch在torch.nn.init中为我们提供了常用的初始化方法，可以发现这些函数除了calculate_gain，所有函数的后缀都带有下划线，意味着这些函数将会直接原地更改输入张量的值
  * 对于不同的损失函数，应该传入不同的增益值，具体是多少官方有提供表格
  * 一般人会定义一个通用的 initialize_weights 方法，在创建模型实例后进行初始化，而不是在模型构造方法内
* 损失函数
  * 损失函数就是模型的负反馈，在PyTorch中，损失函数是必不可少的。它是数据输入到模型当中，产生的结果与真实标签的评价指标，我们的模型可以按照损失函数的目标来做出改进
  * 常用损失函数：
    1. 二分类交叉熵损失函数（分类）
    2. 交叉熵损失函数（分类）
    3. L1损失函数（回归）
    4. MSE损失函数（回归）
    5. 平滑L1 (Smooth L1)损失函数（回归，减轻离群点影响）
    6. 目标泊松分布的负对数似然损失（回归）
    7. KL散度（用于连续分布的距离度量）
    8. MarginRankingLoss（计算两组数据之间的差异）
    9. 多标签边界损失函数（多标签分类）
    10. 二分类损失函数（分类）
    11. 多分类的折页损失（多分类）
    12. 三元组损失（回归）
    13. HingEmbeddingLoss（？）
    14. 余弦相似度（对两个向量做余弦相似度）
    15. CTC损失函数（时序类数据的分类）
* 训练和评估     
  * 训练模型时应切换到训练状态 `model.train()`
  * 测试/使用模型时应使用求值状态`model.eval()`
  * 步骤简述
      ```python
      for data, label in train_loader: # 循环读取DataLoader中的全部数据
        data, label = data.cuda(), label.cuda() # 将数据放到GPU上用于后续计算
        optimizer.zero_grad() # 当前批次数据训练时，先将优化器的梯度置零
        output = model(data) # 将data送入模型中训练    
        loss = criterion(output, label) # 计算损失函数
        loss.backward() # loss反向传播回网络
        optimizer.step() # 使用优化器更新模型参数
      ```
  * 测试/使用模型时
    1. 需要预先设置torch.no_grad，以及将model调至eval模式
    2. 不需要将优化器的梯度置零
    3. 不需要将loss反向回传到网络
    4. 不需要更新optimizer    
* 可视化
  * 可视化是一个可选项，指的是某些任务在训练完成后，需要对一些必要的内容进行可视化，比如分类的ROC曲线，卷积网络中的卷积核，以及训练/验证过程的损失函数曲线等等
* Pytorch优化器
  * 优化器是根据网络反向传播的梯度信息来更新网络的参数，以起到降低loss函数计算值，使得模型输出更加接近真实标签
  * Pytorch提供的优化器：torch.optim，在这里面提供了十种优化算法
  * 优化算法均继承于Optimizer
  * 基类 Optimizer 有有三个属性
    1. defaults：存储的是优化器的超参数
    2. state：参数的缓存
    3. param_groups：管理的参数组
  * Optimizer常用的方法  
    1. zero_grad()：清空所管理参数的梯度
    2. step()：执行一步梯度更新
    3. add_param_group()：添加参数组
    4. load_state_dict() ：加载状态参数字典，可以用来进行模型的断点续训练，继续上次的参数进行训练
    5. state_dict()：获取优化器当前状态信息字典
  * 可以给网络不同的层赋予不同的优化器参数


[第四章 基础实战——FashionMNIST时装分类](https://colab.research.google.com/drive/16fhqMfJchtOSrVaau59rwGjtIyiKg1Jr?usp=sharing)

#### 教程小建议

1. 3.5 模型初始化，torch.nn.init内容 排版有问题，markdown 换行失效