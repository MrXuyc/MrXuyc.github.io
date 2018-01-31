---
layout: post
title: 'tensorflow构建神经网络'
date: 2018-01-10
author: MrXuyc
categories: 技术
cover: '/assets/img/tensorflow/tensorflow1.jpg'
tags: tensorflow
---
> tensorflow构建神经网络

## 构建神经网络结构

```python
# 构建神经网络
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('data', one_hot = True)
# 1层神经元
n_hidden_1 = 256
# 2层神经元
n_hidden_2 = 128
# 灰度图  通道点为1 彩色为3
n_input = 784
# out类别
n_classes = 10

# 绑定模型，用于后续代入值   样本数据
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])

stddev = 0.1
# 高斯初始化 方差0.1  
# w1 规格 784*256  w2 规格256*128   b1 为256个神经元  b2 为128个神经元
weights = {
    "w1" : tf.Variable(tf.random_normal([n_input ,n_hidden_1], stddev = stddev)),
    "w2" : tf.Variable(tf.random_normal([n_hidden_1,n_hidden_2], stddev = stddev)),
    "out" : tf.Variable(tf.random_normal([n_hidden_2, n_classes], stddev = stddev))
}
biases = {
    "b1" : tf.Variable(tf.random_normal([n_hidden_1])),
    "b2" : tf.Variable(tf.random_normal([n_hidden_2])),
    "out" : tf.Variable(tf.random_normal([n_classes]))
}

print("NETWORK READY!")
```

## 函数定义

```python
# 运算  前向传播           数据
def multilayer_perceptron(_X, _weights, _biases):
    # (data*w1 +biases["w1"])   sigmoid函数？ 注意完成后要使用激活函数激活，这里使用sigmoid，一般用ReLU
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(_X, _weights["w1"]), _biases["b1"]))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, _weights["w2"]), _biases["b2"]))
    # 已经是输出 不需要sigmoid函数
    return tf.add(tf.matmul(layer_2, _weights["out"]), _biases["out"])
# 得分值
pred = multilayer_perceptron(x,weights,biases)
# 损失函数    平均值             交叉熵函数                   预期值  真实值
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred))
# 梯度下降，取最优化
optm = tf.train.GradientDescentOptimizer(learning_rate = 0.001).minimize(cost)
# 准确率
corr = tf.equal(tf.argmax(pred ,1), tf.argmax(y, 1))
# boolean 转换 float
accr = tf.reduce_mean(tf.cast(corr, "float"))

init = tf.global_variables_initializer()
```

## 运行阶段

```python
training_epochs = 100
batch_size = 100
display_step = 4

with tf.Session() as sess:
    sess.run(init)

    for epochs in range(training_epochs):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples/batch_size)
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            feeds = {x: batch_xs, y: batch_ys}
            sess.run(optm, feed_dict = feeds)
            avg_cost += sess.run(cost, feed_dict = feeds)
        avg_cost = avg_cost / total_batch
        if(epochs+1) % display_step == 0:
            print("Epoch : %0.d/%03d cost: %.9f" % (epochs,training_epochs,avg_cost))
            feeds = {x: batch_xs, y: batch_ys}
            train_acc = sess.run(accr, feed_dict = feeds)
            print("TRAIN ACCURACY: %.3f" % (train_acc))
            feeds = {x: mnist.test.images, y: mnist.test.labels}
            test_acc = sess.run(accr, feed_dict = feeds)
            print("TEST ACCURACY: %.3f" % (test_acc))
    print("DONE")

```
