# Task3-PyTorch模型定义（第五章）

[教程原文](https://datawhalechina.github.io/thorough-pytorch/%E7%AC%AC%E4%BA%94%E7%AB%A0/index.html)

#### 关键要点

* PyTorch模型定义的方式  
  * 知识回顾
    * Module 类是 torch.nn 模块里提供的一个模型构造类 (nn.Module)，是所有神经⽹网络模块的基类，我们可以继承它来定义我们想要的模型
    * PyTorch模型定义应包括两个主要部分：各个部分的初始化（__init__）；数据流向定义（forward）
  * 基于nn.Module，我们可以通过Sequential，ModuleList和ModuleDict三种方式定义PyTorch模型
  * nn.Sequential
    * 当模型的前向计算为简单串联各个层的计算时， 此类可以通过更加简单的方式定义模型
    * 可以接收一个子模块的有序字典(OrderedDict) 或者一系列子模块作为参数来逐一添加 Module 的实例，⽽模型的前向计算就是将这些实例按添加的顺序逐⼀计算
    * 优点：简单、易读
    * 缺点：会使得模型定义丧失灵活性，比如需要在模型中间加入一个外部输入时就不适合用此类实现  
  * nn.ModuleList
    * ModuleList 接收一个子模块（或层，需属于nn.Module类）的列表作为输入，然后也可以类似List那样进行append和extend操作。同时，子模块或层的权重也会自动添加到网络中来
    * 要特别注意的是，nn.ModuleList 并没有定义一个网络，它只是将不同的模块储存在一起
    * ModuleList中元素的先后顺序并不代表其在网络中的真实位置顺序，需要经过forward函数指定各个层的先后顺序后才算完成了模型的定义
  * nn.ModuleDict
    * 和ModuleList的作用类似，只是ModuleDict能够更方便地为神经网络的层添加名称
  * 三种方法的比较与适用场景
    * Sequential适用于快速验证结果，因为已经明确了要用哪些层，直接写一下就好了，不需要同时写__init__和forward；
    * ModuleList和ModuleDict在某个完全相同的层需要重复出现多次时，非常方便实现，可以”一行顶多行“；
    * 当我们需要之前层的信息的时候，比如 ResNets 中的残差计算，当前层的结果需要和之前层中的结果进行融合，一般使用 ModuleList/ModuleDict 比较方便
* 利用模型块快速搭建复杂网络
  * 对于大部分模型结构（比如ResNet、DenseNet等），虽然模型有很多层， 但是其中有很多重复出现的结构。考虑到每一层有其输入和输出，若干层串联成的”模块“也有其输入和输出，如果我们能将这些重复出现的层定义为一个”模块“，每次只需要向网络中添加对应的模块来构建模型，这样将会极大便利模型构建的过程
  * U-Net简介
    * U-Net是分割 (Segmentation) 模型的杰作，在以医学影像为代表的诸多领域有着广泛的应用。它通过残差连接结构解决了模型学习中的退化问题，使得神经网络的深度能够不断扩展
  * U-Net模型块分析


#### 教程小建议

1. 