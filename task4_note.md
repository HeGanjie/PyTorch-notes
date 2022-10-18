# Task4-PyTorch进阶训练技巧（第六章）

[教程原文](https://datawhalechina.github.io/thorough-pytorch/%E7%AC%AC%E5%85%AD%E7%AB%A0/index.html)

#### 关键要点

* 自定义损失函数
  * 以函数方式定义:可以通过直接以函数定义的方式定义一个损失函数
  * 以类方式定义：更加常用，需要继承自nn.Module；最好全程使用PyTorch提供的张量计算接口，这样就不需要我们实现自动求导功能并且我们可以直接调用cuda
* 动态调整学习率
  * 使用官方scheduler：在使用官方给出的torch.optim.lr_scheduler时，需要将scheduler.step()放在optimizer.step()后面进行使用
  * 自定义scheduler：自定义函数 xxx 来改变optimizer.param_group中lr的值  
* 模型微调-torchvision
  * 数据缺乏的一种解决办法是应用迁移学习
  * 迁移学习的一大应用场景是模型微调。简单来说，就是我们先找到一个同类的别人训练好的模型，把别人现成的训练好了的模型拿过来，换成自己的数据，通过训练调整一下参数
  * PyTorch中提供了许多预训练好的网络模型（VGG，ResNet系列，mobilenet系列......），这些模型都是PyTorch官方在相应的大型数据集训练好的
  * 模型微调的流程
    * 在源数据集(如ImageNet数据集)上预训练一个神经网络模型，即源模型。
    * 创建一个新的神经网络模型，即目标模型。它复制了源模型上除了输出层外的所有模型设计及其参数。我们假设这些模型参数包含了源数据集上学习到的知识，且这些知识同样适用于目标数据集。我们还假设源模型的输出层跟源数据集的标签紧密相关，因此在目标模型中不予采用。
    * 为目标模型添加一个输出⼤小为⽬标数据集类别个数的输出层，并随机初始化该层的模型参数。
    * 在目标数据集上训练目标模型。我们将从头训练输出层，而其余层的参数都是基于源模型的参数微调得到的
  * 使用已有模型结构: 实例化网络时传递 pretrained 参数
  * 训练特定层
    * 通过设置 model.parameters().requires_grad =False 来冻结部分层
    * 没有锁定的层才会被训练
* 模型微调 - timm
  * 是另一个常见的预训练模型库，叫做timm，这个库是由来自加拿大温哥华Ross Wightman创建的。里面提供了许多计算机视觉的SOTA模型，可以当作是torchvision的扩充版本，并且里面的模型在准确度上也较高  
  * timm的安装：1、pip; 2. git+pip
  * 如何查看预训练模型种类
    * 查看所有 `timm.list_models(pretrained=True)`  
    * 查看特定模型的所有种类：`timm.list_models("*densenet*")`
    * 查看模型的具体参数: ` timm.create_model('resnet34',pretrained=True).default_cfg`
  * 使用和修改预训练模型: create_model 后直接调用
  * 查看某一层模型参数: `list(dict(model.named_children())['conv1'].parameters())`
  * 修改模型（将1000类改为10类输出）: `timm.create_model('resnet34',num_classes=10,pretrained=True)`
  * 改变输入通道数（传入单通道的，但是模型需要三通道图）：可通过添加in_chans=1来改变
  * 模型的保存：timm库所创建的模型是torch.model的子类，所以可直接用torch库中内置的模型参数保存和加载的方法
* 半精度训练
