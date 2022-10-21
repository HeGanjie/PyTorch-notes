# Task5-PyTorch可视化（第七章）

[教程原文](https://datawhalechina.github.io/thorough-pytorch/%E7%AC%AC%E4%B8%83%E7%AB%A0/index.html)

#### 关键要点

* 可视化网络结构
  * 主角 torchinfo：主要用于显示我们的模型参数，输入大小，输出大小，模型的整体参数等
  * 使用print打印模型：只能得出基础构件的信息，既不能显示出每一层的shape，也不能显示对应参数量的大小
  * 使用torchinfo可视化网络结构
    * 安装：`... install torchinfo`
    * 使用：`torchinfo.summary(model，input_size: (如 batch_size,channel,h,w))`
    * 输出：树状信息，包括模块信息（每一层的类型、输出shape和参数量）、模型整体的参数量、模型大小、一次前向或者反向传播需要的内存大小等
    * 注意：使用的是colab或者jupyter notebook时，summary()一定是 cell 的返回值，否则我们就要使用`print(summary(...))`来可视化
* CNN可视化
  * 为了更好地理解CNN工作的方式、改进效果，重要的一步是可视化：包括可视化特征是如何提取、提取到的特征的形式以及模型在输入数据上的关注点等
  * CNN卷积核可视化
    * 在PyTorch中可视化卷积核也非常方便，核心在于特定层的卷积核即特定层的模型权重，可视化卷积核就等价于可视化对应的权重矩阵
    * 作用是看模型提取哪些特征
  * CNN特征图可视化方法
    * 输入的原始图像经过每次卷积层得到的数据称为特征图，作用是看模型提取到的特征是什么样子的
    * 在PyTorch中，提供了一个专用的接口使得网络在前向传播过程中能够获取到特征图，叫做hook。* hook 可以这样理解，数据通过网络向前传播，网络某一层我们预先设置了一个钩子，数据传播过后钩子上会留下数据在这一层的样子，读取钩子的信息就是这一层的特征图
    * 实现步骤：
      1. 先实现了一个hook类
      2. 执行模型前，将该hook类的对象注册到要进行可视化的网络的某层中
      3. 执行模型，model在进行前向传播的时候会调用hook的__call__函数，也就是在此时存储了当前层的输入输出。数据一般记录到 list，每次前向传播一次就调用一次 hook，也就是 list 长度会增加1
      4. 展示 hook
  * CNN class activation map可视化方法
    * class activation map （CAM）的作用是判断哪些变量对模型来说是重要的，即在CNN可视化的场景下，判断图像中哪些像素点对预测结果是重要的
    * 如果对重要区域的梯度感兴趣，可以使用在CAM的基础上进一步改进得到的Grad-CAM（以及诸多变种）
    * 相比可视化卷积核与可视化特征图，CAM系列可视化更为直观，能够一目了然地确定重要区域，进而进行可解释性分析或模型优化改进
    * CAM系列操作的实现可以通过开源工具包pytorch-grad-cam来实现    
  * 使用FlashTorch快速实现CNN可视化
    * 不想写代码进行可视化的话，有不少开源工具能够帮助我们快速实现CNN可视化，如 [FlashTorch](https://github.com/MisaOgura/flashtorch)
    * 注意：使用中发现该package对环境有要求，如果下方代码运行报错，请参考[作者给出的配置或者Colab运行环境](https://github.com/MisaOgura/flashtorch/issues/39)
* 使用TensorBoard可视化训练过程
  * 目的：
    1. 通过绘制损失函数曲线来确定训练的终点
    2. debug: 通过可视化其他内容，如输入数据（尤其是图片）、模型结构、参数分布等
  * 主角：TensorBoard
    * 安装：`pip install tensorboardX`，如果使用PyTorch自带的tensorboard工具，则不需要额外安装tensorboard
    * 基本逻辑：将TensorBoard看做一个记录员，它可以记录我们指定的数据，包括模型每一层的feature map，权重，以及训练loss等等。TensorBoard将记录下来的内容保存在一个用户指定的文件夹里，程序不断运行中TensorBoard会不断记录。记录下的内容可以通过网页的形式加以可视化
    * 配置与启动
      * 使用前，需先指定一个文件夹用于保存数据：`writer=tensorboardX.SummaryWriter('./runs')`
      * PyTorch自带的tensorboard: `torch.utils.tensorboard.SummaryWriter`
      * 启动tensorboard(查看记录的数据)：`tensorboard --logdir=/path/to/logs/ --port=xxxx`
    * TensorBoard模型结构可视化:
      ```python3
      writer.add_graph(model, input_to_model = torch.rand(1, 3, 224, 224))
      writer.close()
      ```  
    * TensorBoard图像可视化
      * 当我们做图像相关的任务时，可以方便地将所处理的图片在tensorboard中进行可视化展示
      * 主要使用：`writer.add_images`
    * TensorBoard连续变量可视化
      * 主要用于可视化连续变量（或时序变量）的变化过程；这部分功能非常适合损失函数的可视化
      * 主要使用：`writer.add_scalar`
    * TensorBoard参数分布可视化
      * 当我们需要对参数（或向量）的变化，或者对其分布进行研究时，可以方便地用TensorBoard来进行可视化，主要通过`writer.add_histogram`实现  
    * 服务器端使用TensorBoard
      * 主要是使用 ssh 本地端口转发，让对本地端口的访问请求转发到远程机器的 web 端口

  