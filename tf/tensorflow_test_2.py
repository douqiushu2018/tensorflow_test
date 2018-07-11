"""
<<TensorFlow入门与实战>>
代码3.9 入门
"""
import tensorflow as tf
import numpy as np

# STEP-1 数据来源，构造获取数据函数，参数number为一次获取数据的batch大小
def get_data(number):
    list_x = []
    list_label = []
    for i in range(number):
        x = np.random.randn(1)
        # 数据满足 y = 2 * x + 10
        label = 2 * x + 10 + np.random.randn(1) * 0.01
        list_x.append(x)
        list_label.append(label)
    return list_x, list_label
#  定义训练数据占位符
train_x = tf.placeholder(tf.float32)
train_label = tf.placeholder(tf.float32)
test_x = tf.placeholder(tf.float32)
test_label = tf.placeholder(tf.float32)

# STEP-2 构建网络模型，输入是x，输出是y = weight * x + bias
def interface(x):
    weight = tf.Variable(0.01, name='weight')
    bias = tf.Variable(0.01, name='bias')
    y = weight * x + bias
    return y
# 所以训练值为:
with tf.variable_scope('interface'):
    train_y = interface(train_x)
    # 定义相同名字的变量是共享变量，变量可重用 ??? TODO
    tf.get_variable_scope().reuse_variables()
    test_y = interface(test_x)

# 定义损失函数，这里用差的平方作为损失函数
train_loss = tf.square(train_y - train_label)
test_loss = tf.square(test_y - test_label)
# 定义优化器，这里用梯度下降优化函数
opt = tf.train.GradientDescentOptimizer(0.002)
train_op = opt.minimize(train_loss)

# STEP-3 训练
train_data_x, train_data_label = get_data(1000)
test_data_x, test_data_label = get_data(1)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for i in range(1000):
        sess.run(train_op, feed_dict={train_x: train_data_x[i],
                                      train_label: train_data_label[i]})
        if i % 10 == 0:
            test_loss_value = sess.run(test_loss, feed_dict={test_x: test_data_x[0],
                                                             test_label: test_data_label[0]})
            print('step: %d test_loss_value: %.3f' %(i, test_loss_value))

            # train_loss_value = sess.run(train_loss, feed_dict={train_x: train_data_x[i],
            #                                                    train_label: train_data_label[i]})
            # print('step: %d train_loss_value: %.3f' % (i, train_loss_value))


