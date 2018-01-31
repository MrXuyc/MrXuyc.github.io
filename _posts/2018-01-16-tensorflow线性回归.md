---
layout: post
title: 'tensorflow线性回归'
date: 2018-01-10
author: MrXuyc
categories: 技术
cover: '/assets/img/tensorflow/tensorflow1.jpg'
tags: tensorflow
---
> tensorflow线性回归

## 步骤
1、构造误差为正态分布的数据样本点
2、声明预估值模型 y = Wx + b
3、最小二乘法构造loss函数，即误差  (loss function  其中选择了梯度下降优化参数) loss = tf.reduce_mean(tf.square(y - y_data), name='loss')
4、初始化梯度下降优化器  tf.train.GradientDescentOptimizer
5、使用优化器最小化误差  optimizer.minimize
6、在session中进行训练

```python
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
# 构造数据:1000个点，围绕在 y = 0.1x + 0.3 周围
num_points = 1000
vectors_set = []
for i in range(num_points):
    # 高斯分布  均值为0  标准差为0.55  
    x1 = np.random.normal(0.0, 0.55)
    #               增加随机值  高斯分布  均值为0  标准差为0.03  
    y1 = x1 * 0.1 + 0.3 + np.random.normal(0.0, 0.03)
    vectors_set.append([x1, y1])

# 生成一些样本
x_data = [v[0] for v in vectors_set]
y_data = [v[1] for v in vectors_set]

plt.scatter(x_data, y_data, c='r')

```

![](/assets/img/tensorflow/data/linear/1.jpg)


```python
# 生成参数W，取值为[-1, 1]间随机数
W = tf.Variable(tf.random_uniform([1], -1.0, 1.0), name='W')
# 生成偏移量b
b = tf.Variable(tf.zeros([1]), name='b')
# 经过计算得出预估值y 这里没有用矩阵乘法，是因为都是一维的
y = W * x_data + b

# 预估值y与实际值y_data均方差作为损失函数
loss = tf.reduce_mean(tf.square(y - y_data), name='loss')
# 初始化梯度下降法优化器，用于优化参数，0.5表示学习率，一般设置较小0.001
optimizer = tf.train.GradientDescentOptimizer(0.5)
# 训练过程就是使用优化器，最小化误差值
train = optimizer.minimize(loss, name='train')

# 初始化变量并打印
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print ('W=', W.eval(), "b=", b.eval(), 'loss=', loss.eval())

    # 执行20次训练
    for step in range(20):
        sess.run(train)
        print ('W=', W.eval(), "b=", b.eval(), 'loss=', loss.eval())

    # 展示拟合后的直线图
    plt.scatter(x_data,y_data,c='r')
    plt.plot(x_data, sess.run(W)*x_data+sess.run(b))

```

![](/assets/img/tensorflow/data/linear/2.jpg)
