---
layout: post
title: 'tensorflow基础知识'
date: 2018-01-10
author: MrXuyc
categories: 技术
cover: '/assets/img/tensorflow/tensorflow1.jpg'
tags: tensorflow
---
> tensorflow基础知识

## Tensorflow变量
```
# 完成两个变量的相乘操作
import tensorflow as tf

# 定义tensor类型变量，及乘积操作
w = tf.Variable([[1, 2]])
x = tf.Variable([[2], [3]])
y = tf.matmul(w, x)

# 此时只是定义好了w,x,y的tensor类型的模型/框架，w,x,y并没有值
# <tf.Variable 'Variable_18:0' shape=(1, 2) dtype=float32_ref>
print (w)
# <tf.Variable 'Variable_19:0' shape=(2, 1) dtype=float32_ref>
print (x)
# Tensor("MatMul_8:0", shape=(1, 1), dtype=float32)
print (y)

# tesorflow初始化操作
init_op = tf.global_variables_initializer()

# session表示一次会话，在会话内进行调用
with tf.Session() as sess:
    sess.run(init_op)
    # 需要注意，y是tensor类型的数据，需要使用eval函数查看值 [[ 2.]]
    # 如果使用eval 返回的是模型/框架
    print (y.eval())

```

## Tensorflow基本操作

```
import tensorflow as tf;
tensor = tf.constant(-1.0,shape=[2,3])
# float32 类型会减少问题，尽量使用float32

# 初始化0矩阵
a = tf.zeros([3,4],tf.int32)
# 'tensor’ is [[1,2,3],[4,5,6]]  与tensor格式相同的0矩阵
b = tf.zeros_like(tensor)
# 初始化1矩阵
c = tf.ones([3,4],tf.int32)
#  ‘tensor’ is [[1,2,3],[4,5,6]]  与tensor格式相同的1矩阵(单位矩阵)
d = tf.ones_like(tensor)
# [1 2 3 4 5 6]
tensor = tf.constant([1,2,3,4,5,6])
# [[-1,-1,-1][-1,-1,-1]]
tensor = tf.constant(-1.0,shape=[2,3])
# 包括首尾，取*个 return [ 10.   12.5  15. ]
e = tf.linspace(10.0,15.0,3,name="linspace")
# 包括首 在_1~_2的范围内，以_3 为间隔取
#  return [ 3  6  9 12 15]
f = tf.range(3,18,3);
# mean 均值  stddev 方差
#[[-3.36815619 -1.34553051  1.44139099]
# [ 1.34940624  3.40300703  3.60061264]]
g = tf.random_normal([2,3],mean=-1,stddev=4)
# 创建一个变量
#[[1 2]
# [3 4]
# [5 6]]
h = tf.constant([[1,2],[3,4],[5,6]])
# 洗牌操作  可能跟之前相同
#[[1 2]
# [3 4]
# [5 6]]
i = tf.random_shuffle(h)

init_op=tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init_op)
    print(a.eval())
    print(b.eval())
    print(c.eval())
    print(d.eval())
    print(e.eval())
    print(f.eval())
    print(g.eval())
    print(h.eval())
    print(i.eval())
    print(tensor.eval())

```

## 变量自增操作

```
# 初始变量
state = tf.Variable(0)
# 定义相加操作
new_value = tf.add(state , tf.constant(1))
# 赋值操作将 new_value 赋值给 state
update = tf.assign(state , new_value)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for _ in range(3):
        # 执行相加操作
        sess.run(update)
        print(state.eval())
```

## Saver

```
# saver.save保存  可以调出复用
w = tf.Variable([[0.5,1.0]])
x = tf.Variable([[2.0],[1.0]])
y = tf.matmul(w,x)
init_op = tf.global_variables_initializer()
saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(init_op)
    # 需要创建父级目录
    # 目录内容 test.index  test.meta  test.data  checkpoint
    save_path = saver.save(sess,"C://tensorflow//model//test")
    print( " Model saved in file: ",save_path)

```

## numpy to tensor

```
# tf.convert_to_tensor
import numpy as np
a = np.zeros((3,3))
ta = tf.convert_to_tensor(a)
with tf.Session() as sess:
    print(sess.run(ta))
```

## saver

```
# saver.save保存
w = tf.Variable([[0.5,1.0]])
x = tf.Variable([[2.0],[1.0]])
y = tf.matmul(w,x)
init_op = tf.global_variables_initializer()
saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(init_op)
    # 需要创建父级目录
    # 目录内容 test.index  test.meta  test.data  checkpoint
    save_path = saver.save(sess,"C://tensorflow//model//test")
    print( " Model saved in file: ",save_path)

```

## 占位操作

```
# 在session中占位
input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)
# 乘法操作
output = tf.multiply(input1,input2)
with tf.Session() as session:
    # [array([ 14.], dtype=float32)]
    print(session.run([output],feed_dict={input1:[7.],input2:[2.]}))
```
