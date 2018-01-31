---
layout: post
title: 'tensorflow的MNIST数据集'
date: 2018-01-10
author: MrXuyc
categories: 技术
cover: '/assets/img/tensorflow/tensorflow1.jpg'
tags: tensorflow
---
> tensorflow的MNIST数据集

```python
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
#import input_data
print("packs loaded")

```

```python
print("Download and Extract MNIST dataset")
# 目录
mnist = input_data.read_data_sets('data/' ,one_hot = True)
print(" type of 'mnist' is %s" % (type(mnist)))
print(" number of train data is %d" % (mnist.train.num_examples))
print(" number of test data is %d" %  (mnist.test.num_examples))
# 查看数据shape：训练集、测试集、验证集
print (mnist.train.images.shape, mnist.train.labels.shape)
print (mnist.test.images.shape, mnist.test.labels.shape)
print (mnist.validation.images.shape, mnist.validation.labels.shape)
```
![](/assets/img/tensorflow/data/mnist/2.jpg)

```python
# 在训练集中随机取出五条数据的索引
index = np.random.randint(mnist.train.images.shape[0], size=5)
trainimg = mnist.train.images
trainlabel = mnist.train.labels

for i in index:
    img = np.reshape(trainimg[i, :], (28, 28))
    label = np.argmax(trainlabel[i, :])
    plt.matshow(img, cmap=plt.get_cmap('gray'))
    plt.title('index:' + str(i) + ',label:' + str(label))
    plt.show()
```
![](/assets/img/tensorflow/data/mnist/1.jpg)

```python
# 训练神经网络时，一个batch一个batch进行
batch_size = 100
batch_xs, batch_ys = mnist.train.next_batch(batch_size)
print (type(batch_xs))
print (type(batch_ys))
print (batch_xs.shape)
print (batch_ys.shape)
```
![](/assets/img/tensorflow/data/mnist/3.jpg)
