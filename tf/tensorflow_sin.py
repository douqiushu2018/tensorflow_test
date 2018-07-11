#coding=utf-8

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pylab

# 标准sin函数图，用于对比
def draw_correct_line():
    x = np.arange(0, 2 * np.pi, 0.01)
    x = x.reshape((len(x), 1))
    y = np.sin(x)
    pylab.plot(x, y, label='标准sin曲线')
    plt.axhline(linewidth=1, color='r')

# 构造训练样本
def get_train_data():
    train_x = np.random.uniform(0.0, 2 * np.pi, (1))
    train_y = np.sin(train_x)
    return train_x, train_y

# 定义网络结构
def interface(input_data):
    with tf.variable_scope('hidden1'):
        weight = tf.get_variable('weight', [1, 16], tf.float32,
                                 initializer=tf.random_normal_initializer(0.0, 1))
        bias = tf.get_variable('bias', [1, 16], tf.float32,
                               initializer=tf.random_normal_initializer(0.0, 1))
        # 激活函数，值将传入下一层
        hidden1 = tf.sigmoid(tf.multiply(input_data, weight) + bias)

    with tf.variable_scope('hidden2'):
        weight = tf.get_variable('weight', [16, 16], tf.float32,
                                 initializer=tf.random_normal_initializer(0.0, 1))
        bias = tf.get_variable('bias', [16], tf.float32,
                               initializer=tf.random_normal_initializer(0.0, 1))
        hidden2 = tf.sigmoid(tf.matmul(hidden1, weight) + bias)

    with tf.variable_scope('hidden3'):
        weight = tf.get_variable('weight', [16, 16], tf.float32,
                                 initializer=tf.random_normal_initializer(0.0, 1))
        bias = tf.get_variable('bias', [16], tf.float32,
                               initializer=tf.random_normal_initializer(0.0, 1))
        hidden3 = tf.sigmoid(tf.matmul(hidden2, weight) + bias)

    with tf.variable_scope('output_layer'):
        weight = tf.get_variable('weight', [16, 1], tf.float32,
                                 initializer=tf.random_normal_initializer(0.0, 1))
        bias = tf.get_variable('bias', [1], tf.float32,
                               initializer=tf.random_normal_initializer(0.0, 1))
        output = tf.matmul(hidden3, weight) + bias
    return output

# 定义训练流程
def train():
    leaning_rate = 0.01

    x = tf.placeholder(tf.float32)
    y = tf.placeholder(tf.float32)

    net_out = interface(x)

    # 损失函数 - 差的平方
    less_op = tf.square(net_out - y)

    # 随机梯度函数 - 学习率
    opt = tf.train.GradientDescentOptimizer(leaning_rate)
    train_op = opt.minimize(less_op)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        print('train start...')

        for i in range(1000000):
            train_x, train_y = get_train_data()
            sess.run(train_op, feed_dict={x: train_x, y: train_y})

            if i % 10000 == 0:
                times = int(i / 1000)
                test_x_ndarray = np.arange(0, 2 * np.pi, 0.01)
                test_y_ndarray = np.zeros([len(test_x_ndarray)])
                ind = 0
            for test_x in test_x_ndarray:
                test_y = sess.run(net_out, feed_dict={x: test_x, y: 1})
                print(test_y_ndarray.shape, 'test_y_ndarray.shape')
                print(test_y.shape, 'test_y.shape')
                np.put(test_y_ndarray, ind, test_y)
                ind += 1
            draw_correct_line()
            #print(test_x_ndarray, 'test_x_ndarray')
            #print(test_y_ndarray, 'test_y_ndarray')
            pylab.plot(test_x_ndarray, test_y_ndarray, '--', label=str(times) + 'times')
            print('plot...')
            pylab.show()
            print('show...')
        print('train end...')

if __name__ == "__main__":
    train()