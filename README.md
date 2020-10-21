# SiameseNet

## 前言

对于人来说，你会玩CS就很容易上手CF。但是对于神经网络来说，并非如此。如果让一个汽车分类网络去识别不同的单车，那效果肯定很差。而传统的CNN网络都是输入大量的数据，然后进行分类的学习。但是这样做的问题就是，神经网络的通用性太差了，根本达不到“智能”的标准。人类的认知系统，可以通过少量的数据就可以从中学习到规律。

所以为了模拟人类的大脑，Gregory Koch提出了Siamese Neural Networks译为暹罗网络。因为暹罗是人类历史上第一次注意到的连体人病症，所以也以暹罗命名，大家把他叫成孪生网络就好了。

## 原理

其实对于小样本的数据来说，我们要让网络学习的是分辨数据相似之处的能力。所以现在假设有两个图片，我们将他输入到两个网络中，提取出一个特征向量。之后对这两个向量进行L1距离的计算，如果是同一类的图片，我们希望这个差值越小越好。如果是不同类的图片自然希望它越大越好。

那么实际上这两个网络是可以共享权值的，因为提取的特征都是一样的。所以本质上他是一个网络，可以接收两个输入而已。也有不共享权值的网络，但在这里就不加以讨论

![68747470733a2f2f736f72656e626f756d612e6769746875622e696f2f696d616765732f5369616d6573655f6469616772616d5f322e706e67.png](https://i.loli.net/2020/05/20/AwIHKXkgsQNGqrF.png)

## 安装

```python
$ git clone https://github.com/Runist/SiameseNet
$ cd SiameseNet
$ sudo pip install -r requirements.txt
```

## 数据集

在这个小Demo中用到的数据集是各个不同语言的手写字体，每个语言中挑选了几个字母，分别让20个人进行手写，所以每个字母对应有20个不同样式。

## 使用

这个Demo主要用keras编写，以Tensorflow2.0作为后端，用到了少许Tensorflow2.0的代码。config.py中可以对参数进行调节，其中要说明的是train_mode可以分为用generator和on_batch训练。generator是用fit_generator，可以自己添加callbacks参数，而on_batch则是一个一个batch训练，可以自行收集训练过程的参数。

在训练之前，需要运行load_data.py将数据写入到.pickle文件中，或者将dataReader.py中的from load_data import *反注释即可。