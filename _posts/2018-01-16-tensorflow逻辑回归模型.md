---
layout: post
title: 'tensorflow逻辑回归模型'
date: 2018-01-10
author: MrXuyc
categories: 技术
cover: '/assets/img/tensorflow/tensorflow1.jpg'
tags: tensorflow
---
> tensorflow逻辑回归模型

```
#逻辑回归
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

# 读取mnist文件 并且帮助预处理
mnist = input_data.read_data_sets("data",one_hot=True)
trainimg = mnist.train.images
trainlabel = mnist.train.labels
testimg = mnist.test.images
testlabel = mnist.test.labels
print("MNIST DATA LOADED")

# 28*28  784像素点
print(trainimg.shape)
# 10个指标   10个数字[0,0,0,0,0,0,1,0,0,0]
print(trainlabel.shape)
print(testimg.shape)
print(testlabel.shape)

# 构建规格（纬度和类型）  None 大小   不给实际值
x = tf.placeholder("float",[None,784])
y = tf.placeholder("float",[None,10])
# 10为输出
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
# 输入的分支   Wx+b   预测值结果
actv = tf.nn.softmax(tf.matmul(x, W) + b)
# cost函数   -logP   P 为将上述得分进行归一化操作，属于哪个类别的概率值
#  tf.log(actv)正确类别的概率值  y * tf.log(actv)  [0,0,0,0,0,0,1,0,0,0] * tf.log(actv)
cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(actv),reduction_indices = 1))

learning_rate = 0.01
# 梯度下降算法
optm = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# agemax   求出预测值最大的     
pred = tf.equal(tf.argmax(actv,1),tf.argmax(y,1))
# cast  pred true?false?  转换类型  reduce_mean求均值
accr = tf.reduce_mean(tf.cast(pred,"float"))

init = tf.global_variables_initializer()

# #argmax方法测试
# sess = tf.InteractiveSession()
# arr = np.array([[31,23,4,24,27,34],[18,3,25,0,6,35],[28,14,33,22,20,8],
#                [13,30,21,19,7,9],[16,1,26,32,2,29],[17,12,5,11,10,15],])
# # rank 当前的矩阵纬度
# print(tf.rank(arr).eval())
# # shape 几排几列
# print(tf.shape(arr).eval())
# # 最大值的索引   0为列比较  1为行比较
# print(tf.argmax(arr,0).eval())
# print(tf.argmax(arr,1).eval())


# 样本迭代次数
training_epochs = 1000
# 每次迭代多少个样本
batch_size = 100
display_step = 5
with tf.Session() as sess:
    # 初始化
    sess.run(init)
    for epoch in range(training_epochs):
        # 损失值
        avg_cost = 0
        num_batch = int(mnist.train.num_examples/batch_size)
        for i in range(num_batch):
            batch_xs,batch_ys = mnist.train.next_batch(batch_size)
            sess.run(optm,feed_dict = {x:batch_xs, y:batch_ys})
            feeds = {x: batch_xs, y:batch_ys}
            avg_cost += sess.run(cost, feed_dict=feeds)/num_batch

        if epoch % display_step ==0:
            feeds_train = {x: batch_xs, y:batch_ys}
            feeds_test = {x: mnist.test.images, y: mnist.test.labels}
            train_acc = sess.run(accr, feed_dict=feeds_train)
            test_acc = sess.run(accr,feed_dict=feeds_test)
            print("Epoch: %03d/%03d cost: %.9f train_acc: %.3f test_acc: %.3f"
                 % (epoch,training_epochs,avg_cost,train_acc,test_acc))

    print("DONE")
```
