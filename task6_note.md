# Task06-PyTorch生态简介+模型部署（第八、九章）

[教程原文](https://datawhalechina.github.io/thorough-pytorch/%E7%AC%AC%E5%85%AB%E7%AB%A0/index.html)

#### 关键要点

* PyTorch 生态
  * 即开源社区围绕PyTorch所产生的一系列工具包，例如：
  * 计算机视觉：TorchVision、TorchVideo
  * 自然语言处理：torchtext，HuggingFace
  * 图卷积网络: PyTorch Geometric
* TorchVision
  * 简介：包含了在计算机视觉中常常用到的数据集，模型和图像处理的方式。帮助我们解决了常见的计算机视觉中一些重复且耗时的工作，并在数据集的获取、数据增强、模型预训练等方面大大降低了我们的工作难度，可以让我们更加快速上手一些计算机视觉任务
  * 常用的类：dataset、models、tramsforms
  * torchvision.tramsforms
    * 主要包含了一些我们在计算机视觉中常见的数据集
  * torchvision.transforms
    * 用于图片大小归一化、缩放或翻转等变换等操作，操作可以通过`transforms.Compose`合并
    * [实战课程](https://datawhalechina.github.io/thorough-pytorch/%E7%AC%AC%E5%85%AB%E7%AB%A0/transforms%E5%AE%9E%E6%93%8D.html)
  * torchvision.models
    * 一些预训练好的模型，可以分为这几类：
      1. Classification（分类）
      2. Semantic Segmentation（语义分割）
      3. Object Detection，instance Segmentation and Keypoint Detection（物体检测，实例分割和人体关键点检测）
      4. Video classification（视频分类）
  * torchvision.io
    * 提供了视频、图片和文件的 IO 操作的功能，它们包括读取、写入、编解码处理操作
    * 注意：
      1. 不同版本的 API 差别较大    
      2. 除了read_video()等方法，此类提供了一个细粒度的视频API VideoReader() ，它效率更高且更加接近底层处理。使用时，需先安装ffmpeg然后从源码重新编译torchvision
      3. 使用Video相关API时，最好提前安装PyAV
  * torchvision.ops
    * 提供了许多计算机视觉的特定操作，包括但不仅限于NMS，RoIAlign（MASK R-CNN中应用的一种方法），RoIPool（Fast R-CNN中用到的一种方法）。在合适的时间使用可以大大降低我们的工作量，避免重复的造轮子
  * torchvision.utils
    * 提供了一些可视化的方法，可以帮助我们将若干张图片拼接在一起、可视化检测和分割的效果  
* [PyTorchVideo](https://pytorchvideo.readthedocs.io/en/latest/index.html)
  * 使用方法与torchvision类似，是一个专注于视频理解工作的深度学习库
  * 提供了加速视频理解研究所需的可重用、模块化和高效的组件。
  * 是用PyTorch开发的，支持不同的深度学习视频组件，如视频模型、视频数据集和视频特定转换
  * 主要部件和亮点
    * 提供了加速视频理解研究所需的模块化和高效的API
    * 支持不同的深度学习视频组件，如视频模型、视频数据集和视频特定转换
    * 提供了model zoo，使得人们可以使用各种先进的预训练视频模型及其评判基准
    * 亮点：
      1. 基于 PyTorch
      2. Model Zoo: 提供了包含I3D、R(2+1)D、SlowFast、X3D、MViT等SOTA模型的高质量model zoo，并且与PyTorch Hub做了整合，大大简化模型调用
      3. 数据预处理和常见数据，支持Kinetics-400, Something-Something V2, Charades, Ava (v2.2), Epic Kitchen, HMDB51, UCF101, Domsev等主流数据集和相应的数据预处理，同时还支持randaug, augmix等数据增强trick
      4. 模块化设计：库的设计类似于torchvision，提供了许多模块方便用户调用修改，具体包括data, transforms, layer, model, accelerator等模块，方便用户进行调用和读取
      5. 支持多模态：支持包括了visual和audio，未来会支持更多模态
      6. 移动端部署优化：支持针对移动端模型的部署优化
  * 安装：`pip install pytorchvideo`
    * 注意：
      1. python版本 >= 3.7
      2. PyTorch >= 1.8.0，安装的torchvision也需要匹配    
      3. CUDA >= 10.2
      4. ioPath 和 fvcore 请根据官方文档判断
  * Model zoo
    * benchmark：略，请看[原文](https://datawhalechina.github.io/thorough-pytorch/%E7%AC%AC%E5%85%AB%E7%AB%A0/8.3%20%E8%A7%86%E9%A2%91%20-%20PyTorchVideo.html)
    * 如何使用，官方提供了三种使用方法: 
      * TorchHub，这些模型都已经在TorchHub存在。可以根据实际情况来选择需不需要使用预训练模型。[tutorial](https://pytorchvideo.org/docs/tutorial_torchhub_inference)
      * PySlowFast，[官网](https://github.com/facebookresearch/SlowFast/)
      * PyTorch Lightning，[官网](https://github.com/PyTorchLightning/pytorch-lightning)
      * [更多使用教程](https://github.com/facebookresearch/pytorchvideo/tree/main/tutorials)
* TorchText
  * 是PyTorch官方用于自然语言处理（NLP）的工具包
  * 可以方便的对文本进行预处理，例如截断补长、构建词表等
  * 主要组成部分：
    * 数据处理工具 torchtext.data.functional、torchtext.data.utils
    * 数据集 torchtext.data.datasets
    * 词表工具 torchtext.vocab
    * 评测指标 torchtext.metrics
  * 安装: `pip install torchtext`  
  * 构建数据集:
    * Field及其使用
      * Field是torchtext中定义数据类型以及转换为张量的指令
      * 出现原因：torchtext 认为一个样本是由多个字段（文本字段，标签字段）组成，不同的字段可能会有不同的处理方式
      * 定义Field对象是为了明确如何处理不同类型的数据，但具体的处理则是在Dataset中完成的
      * 使用方式：略，请看[原文](https://datawhalechina.github.io/thorough-pytorch/%E7%AC%AC%E5%85%AB%E7%AB%A0/8.4%20%E6%96%87%E6%9C%AC%20-%20torchtext.html)
    * 词汇表（vocab）
      * 将句子中的词语转为向量表示
      * 可以使用Field自带的build_vocab函数完成词汇表构建 `TEXT.build_vocab(train)`
    * 数据迭代器:
      * 其实就是torchtext中的DataLoader，具体请看原文代码  
      * 支持只对一个dataset和同时对多个dataset构建数据迭代器
    * 使用自带数据集: torchtext也提供若干常用的数据集方便快速进行算法测试，具体请看[官方文档](https://pytorch.org/text/stable/datasets.html)
  * 评测指标（metric）
    * NLP中部分任务的评测不是通过准确率等指标完成的，比如机器翻译任务常用BLEU (bilingual evaluation understudy) score来评价预测文本和标签文本之间的相似程度
    * torchtext中可以直接调用torchtext.data.metrics.bleu_score来快速实现BLEU
  * 其他
    1. 注意：NLP常用的网络结构比较固定，主要通过torch.nn中的模块来实现，比如torch.nn.LSTM、torch.nn.RNN等
    2. 对于文本研究而言，当下Transformer已经成为了绝对的主流，因此PyTorch生态中的[HuggingFace](https://huggingface.co/)等工具包也受到了越来越广泛的关注  
* PyTorch的模型部署
  * 使用ONNX进行部署并推理
    * 通常人们会将模型部署在手机端、开发板，嵌入式设备上，但是这些设备上由于框架的规模，环境依赖，算力的限制，我们无法直接使用训练好的权重进行推理，因此我们需要将得到的权重进行变换才能使我们的模型可以成功部署在上述设备上
  * ONNX和ONNX Runtime简介
    * [ONNX( Open Neural Network Exchange)](https://github.com/onnx/onnx) 是 Facebook (现Meta) 和微软在2017年共同发布的，用于标准描述计算图的一种格式
    * ONNX通过定义一组与环境和平台无关的标准格式，使AI模型可以在不同框架和环境下交互使用，它可以看作深度学习框架和部署端的桥梁，就像编译器的中间语言一样
    * 由于各框架兼容性不一，我们通常只用 ONNX 表示更容易部署的静态图。硬件和软件厂商只需要基于ONNX标准优化模型性能，让所有兼容ONNX标准的框架受益
    * 目前，ONNX主要关注在模型预测方面，使用不同框架训练的模型，转化为ONNX格式后，可以很容易的部署在兼容ONNX的运行环境中
    * 目前，在微软，亚马逊 ，Facebook(现Meta) 和 IBM 等公司和众多开源贡献的共同维护下，ONNX 已经对接了下图的多种深度学习框架和多种推理引擎  
    * [ONNX Runtime](https://github.com/microsoft/onnxruntime)简介
      * 是由微软维护的一个跨平台机器学习推理加速器，可直接读取.onnx文件并实现推理
      * PyTorch借助ONNX Runtime也完成了部署的最后一公里，构建了 PyTorch --> ONNX --> ONNX Runtime 部署流水线，我们只需要将模型转换为 .onnx 文件，并在 ONNX Runtime 上运行模型即可
  * 安装: pip install onnx、onnxruntime、onnxruntime-gpu（使用GPU进行推理）
  * 注意：
    1. [ONNX和ONNX Runtime之间的适配关系](https://github.com/microsoft/onnxruntime/blob/master/docs/Versioning.md)
    2. 想使用GPU进行推理时，我们需要先将安装的onnxruntime卸载，再安装onnxruntime-gpu，还需要考虑[ONNX Runtime与CUDA之间的适配关系](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)
  * 模型导出为ONNX
    * 使用torch.onnx.export()把模型转换成 ONNX 格式的函数
    * 注意：必须调用model.eval()或者model.train(False)以确保我们的模型处在推理模式下
  * ONNX模型的检验
    * 通过onnx.checker.check_model()进行检验，具体看[原文](https://datawhalechina.github.io/thorough-pytorch/%E7%AC%AC%E4%B9%9D%E7%AB%A0/9.1%20%E4%BD%BF%E7%94%A8ONNX%E8%BF%9B%E8%A1%8C%E9%83%A8%E7%BD%B2%E5%B9%B6%E6%8E%A8%E7%90%86.html#id5)
  * ONNX可视化
    * [Netron](https://github.com/lutzroeder/netron)可以像Tensorboard一样可视化模型来观察每个节点的属性特征，实现onnx的可视化
  * 使用ONNX Runtime进行推理
    * 使用ONNX Runtime运行一下转化后的模型，看一下推理后的结果，具体看代码[原文](https://datawhalechina.github.io/thorough-pytorch/%E7%AC%AC%E4%B9%9D%E7%AB%A0/9.1%20%E4%BD%BF%E7%94%A8ONNX%E8%BF%9B%E8%A1%8C%E9%83%A8%E7%BD%B2%E5%B9%B6%E6%8E%A8%E7%90%86.html#id7)
    * 注意：
      1. ONNX的输入不是tensor而是array，因此我们要对张量进行变换或者直接将数据读取为array格式
      2. 输入的array的shape应该和我们导出模型的dummy_input的shape相同，如果图片大小不一样，我们应该先进行resize
      3. run的结果是一个列表，我们需要进行索引操作才能获得array格式的结果
      4. 在构建输入的字典时，我们需要注意字典的key应与导出ONNX格式设置的input_name相同，因此我们更建议使用`ort_session.get_inputs()[0].name`读取 name 而不是硬编码字符串
  * PyTorch官网示例[ONNX实战代码](https://datawhalechina.github.io/thorough-pytorch/%E7%AC%AC%E4%B9%9D%E7%AB%A0/9.1%20%E4%BD%BF%E7%94%A8ONNX%E8%BF%9B%E8%A1%8C%E9%83%A8%E7%BD%B2%E5%B9%B6%E6%8E%A8%E7%90%86.html#id8)
      

#### 教程小建议

1. 8.2.5 从源码重新编译torchvision我们才能我们能使用这些方法；这句出现了两次“我们” 和 “能”

  