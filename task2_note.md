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
  * PyTorch常用模块：torch、torch.nn、torch.utils.data.Dataset、torch.utils.data.DataLoader、torch.optimizer
  * 其他常用库：os、numpy、pandas、matplotlib、seaborn、sklearn
  * 常用超参数：batch_size、lr、max_epochs 
  * 其他：配置启用 cuda

* 数据读入
  * 是通过Dataset+DataLoader的方式完成的，Dataset定义好数据的格式和数据变换形式，DataLoader用iterative的方式不断读入批次数据
  * 可以定义自己的Dataset类来实现灵活的数据读取，定义的类需要继承PyTorch自身的Dataset类
* 模型构建
  * PyTorch中神经网络构造一般是基于 Module 类的模型来完成的，它让模型构造更加灵活
  * ...  

