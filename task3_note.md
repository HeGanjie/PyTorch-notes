# Task3-PyTorch模型定义（第五章）

[教程原文](https://datawhalechina.github.io/thorough-pytorch/%E7%AC%AC%E4%BA%94%E7%AB%A0/index.html)

#### 关键要点

* PyTorch模型定义的方式  
  * 知识回顾
    * Module 类是 torch.nn 模块里提供的一个模型构造类 (nn.Module)，是所有神经⽹网络模块的基类，我们可以继承它来定义我们想要的模型
    * PyTorch模型定义应包括两个主要部分：各个部分的初始化（`__init__`）；数据流向定义（`forward`）
  * 基于nn.Module，我们可以通过Sequential，ModuleList和ModuleDict三种方式定义PyTorch模型
  * nn.Sequential
    * 当模型的前向计算为简单串联各个层的计算时， 此类可以通过更加简单的方式定义模型
    * 可以接收一个子模块的有序字典(OrderedDict) 或者一系列子模块作为参数来逐一添加 Module 的实例，⽽模型的前向计算就是将这些实例按添加的顺序逐⼀计算
    * 优点：简单、易读
    * 缺点：会使得模型定义丧失灵活性，比如需要在模型中间加入一个外部输入时就不适合用此类实现  
  * nn.ModuleList
    * ModuleList 接收一个子模块（或层，需属于nn.Module类）的列表作为输入，然后也可以类似List那样进行append和extend操作。同时，子模块或层的权重也会自动添加到网络中来
    * 特别注意：nn.ModuleList 并没有定义网络，它只是将不同的模块储存在一起，元素的先后顺序并不代表其在网络中的真实顺序，需要经过 forward 函数指定各个层的先后顺序后才算完成了模型的定义
  * nn.ModuleDict
    * 和 ModuleList 类似，只是 ModuleDict 能够更方便地为神经网络的层添加名称
  * 三种方法的比较与适用场景
    * Sequential适用于快速验证结果，因为已经明确了要用哪些层，直接写一下就好了，不需要同时写 `__init__` 和 `forward`；
    * ModuleList和ModuleDict在某个完全相同的层需要重复出现多次时，非常方便实现，可以”一行顶多行“；
    * 当我们需要之前层的信息的时候（如 ResNets 中的残差计算），当前层的结果需要和之前层中的结果进行融合，一般使用 ModuleList/ModuleDict 比较方便
* 利用模型块快速搭建复杂网络
  * 对于大部分模型结构（比如ResNet、DenseNet等），虽然模型有很多层， 但是其中有很多重复出现的结构。考虑到每一层有其输入和输出，若干层串联成的”模块“也有其输入和输出，如果我们能将这些重复出现的层定义为一个”模块“，每次只需要向网络中添加对应的模块来构建模型，这样将会极大便利模型构建的过程
  * U-Net简介
    * U-Net是分割 (Segmentation) 模型的杰作，在以医学影像为代表的诸多领域有着广泛的应用。它通过残差连接结构解决了模型学习中的退化问题，使得神经网络的深度能够不断扩展
  * U-Net模型块分析
    * 具有非常好的对称性
    * 模型从上到下分为若干层，每层由左侧和右侧两个模型块组成，每侧的模型块与其上下模型块之间有连接
    * 位于同一层左右两侧的模型块之间也有连接，称为“Skip-connection”
    * 每个子块内部的两次卷积（Double Convolution）
    * 左侧模型块之间的下采样连接，即最大池化（Max pooling）
    * 右侧模型块之间的上采样连接（Up sampling）
    * 输出层的处理
  * U-Net模型块实现
    * 相比把每一层按序排列显式写出，更好的是先定义好模型块，再定义模型块之间的连接顺序和计算方式。就好比装配零件一样，我们先装配好一些基础的部件，之后再用这些可以复用的部件得到整个装配体
    * 代码略，请看[原文](https://datawhalechina.github.io/thorough-pytorch/%E7%AC%AC%E4%BA%94%E7%AB%A0/5.2%20%E5%88%A9%E7%94%A8%E6%A8%A1%E5%9E%8B%E5%9D%97%E5%BF%AB%E9%80%9F%E6%90%AD%E5%BB%BA%E5%A4%8D%E6%9D%82%E7%BD%91%E7%BB%9C.html)
* PyTorch修改模型
  * 随着深度学习的发展和PyTorch越来越广泛的使用，有越来越多的开源模型可以供我们使用，很多时候我们也不必从头开始构建模型；如果有一个现成的模型，但该模型中的部分结构不符合我们的要求，为了使用模型，我们可以对模型结构进行必要的修改
  * 修改模型层
    * 关键代码 `net.fc = nn.Sequential(...)`
  * 添加外部输入
    * 基本思路是：将原模型添加输入位置前的部分作为一个整体，同时在forward中定义好原模型不变的部分、添加的输入和后续层之间的连接关系，从而完成模型的修改
    * 通过torch.cat 可以实现tensor的拼接
  * 添加额外输出
    * 有时候在模型训练中，除了模型最后的输出外，我们需要输出模型某一中间层的结果，以施加额外的监督，获得更好的中间层结果。基本的思路是修改模型定义中forward函数的return变量
    * 关键代码 `def forward(...): return x10, x1000` 
* PyTorch模型保存与读取
  * 存储格式：主要采用pkl，pt，pth三种格式，均支持模型权重和整个模型的存储，就使用层面来说没有区别
  * 模型主要包含两个部分：模型结构和权重。其中模型是继承nn.Module的类，权重的数据结构是一个字典（key是层名，value是权重向量）
  * 存储也由此分为两种形式：
    1. 存储整个模型（包括结构和权重）: `torch.save(model, save_dir)`
    2. 只存储模型权重：`torch.save(model.state_dict, save_dir)`
  * 单卡和多卡模型存储的区别
    * 如果要使用多卡训练的话，需要对模型使用torch.nn.DataParallel
    * 差别在于多卡并行的模型每层的名称前多了一个“module”
    * 单卡保存
      * 单卡加载：如果是加载模型权重，则读取时需要先实例化模型，再设置 state_dict 
      * 多卡加载：读取单卡保存的模型后，使用nn.DataParallel函数进行分布式训练设置即可
    * 多卡保存
      * 单卡加载
        1. 对于加载整个模型，直接提取模型的module属性即可
        2. 对于加载模型权重，推荐做法是，直接往model里添加module（对单卡使用 DataParallel）
      * 多卡加载
        * 保存整个模型时会同时保存所使用的GPU id等信息，读取时若这些信息和当前使用的GPU信息不符则可能会报错或者程序不按预定状态运行 
        * 相比之下，读取模型权重，之后再使用nn.DataParallel进行分布式训练设置则没有问题。因此多卡模式下建议使用权重的方式存储和读取模型
        * 如果只有保存的整个模型，也可以采用提取权重的方式构建新的模型

