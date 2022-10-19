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
  * 半精度能够减少显存占用，使得显卡可以同时加载更多数据进行计算
  * 半精度训练的设置
    1. `import autocast`
    2. 模型设置，forward 的定义加上 `@autocast()` 装饰器
    3. 训练过程:在将数据输入模型及其之后的部分放入`with autocast():`
  * 注意：半精度训练主要适用于数据本身的size比较大（比如说3D图像、视频等）。当数据本身的size并不大时（比如手写数字MNIST数据集的图片尺寸只有28*28），使用半精度训练则可能不会带来显著的提升
* 数据增强-imgaug
  * 这类技术，可提高训练数据集的大小和质量，以便我们可以使用它们来构建更好的深度学习模型
  * 在计算视觉领域，生成增强图像相对容易。即使引入噪声或裁剪图像的一部分，模型仍可以对图像进行分类
  * imgaug简介和安装
    * imgaug是计算机视觉任务中常用的一个数据增强的包，相比于torchvision.transforms，它提供了更多的数据增强方法
    * imgaug的安装: [官网](https://github.com/aleju/imgaug)
    * imgaug的使用:
      * imgaug仅仅提供了图像增强的一些方法，但是并未提供图像的IO操作
      * 建议使用imageio进行读入，如果使用的是opencv进行文件读取的时候，需要进行手动改变通道，将读取的BGR图像转换为RGB图像
      * 当我们用PIL.Image进行读取时，因为读取的图片没有shape的属性，所以我们需要将读取到的img转换为np.array()的形式再进行处理
      * 常用方法，先引入 `from imgaug import augmenters as iaa`
        1. 创建旋转处理函数 `rotate = iaa.Affine(rotate=(-4,45))`
        2. 单图多个函数依次处理：`iaa.Sequential([...])`
        3. 一次处理多张图片：`rotate(images=images)`
        4. 随机选择处理方式：`iaa.Sometimes(p,then_list,else_list)`
      * 对不同大小的图片进行处理：不同大小的需要分批处理
    * imgaug在PyTorch的应用：主要用于自定义数据集的 `__getitem__`，[更多说明](https://github.com/aleju/imgaug/issues/406)
    * num_workers>0 时需要注意worker_init_fn()函数的作用
    * 作者还推荐另一个数据增强库:Albumentations
* 使用argparse进行调参
  * argparse可以解析我们输入的命令行参数
  * argparse简介
    * 是python内置的命令行解析的标准模块
    * 使用后，在命令行输入的参数就可以以这种形式 `python file.py --lr 1e-4 --batch_size 32`
  * argparse的使用，可以归纳为三个步骤
    * 创建ArgumentParser()对象
    * 调用add_argument()方法添加参数
    * 使用parse_args()解析参数
  * 必填参数：给参数设置required =True后，我们就必须传入该参数  
  * 更加高效使用argparse修改超参数：为了使代码更加简洁和模块化，一般会将有关超参数的操作写在config.py，然后在其他文件导入


#### 教程小建议
1. 6.5 数据增强，对批次图片进行处理那一小节，有错别字：“将image改为image”应该是“将image改为images”吧