'''
<<TensorFlow 技术解析与实战>>
第八章 第一个TensorFlow程序
'''

import tensorflow as tf
import numpy as np

# STEP-1 生成及加载数据
# -1 ~ 1 之间，300个数，转成300 * 1 的二维数组
x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
# 噪音点，与x_data维度一致，正态分布，均值为0，方差为0.05
noise = np.random.normal(0, 0.05, x_data.shape)
# y = x^2 - 0.5 + 噪音
y_data = np.square(x_data) - 0.5 + noise
# 定义占位符
xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])

# STEP-2 构建网络模型
def add_layer(inputs, in_size, out_size, activation_function=None):
    # 构建权重：in_size * out_size 大小的矩阵
    weights = tf.Variable(tf.random_normal([in_size, out_size]))
    # 构建偏置：1 * out_size 大小的矩阵
    biases = tf.Variable(tf.random_normal([1, out_size]) + 0.1)
    # 矩阵相乘
    Ws_plus_b = tf.matmul(inputs, weights) + biases
    print(inputs.shape, 'inputs')
    print(weights.shape, 'weights')
    print(Ws_plus_b.shape, 'Ws_plus_b')
    print('===============')
    if activation_function is None:
        outputs = Ws_plus_b
    else:
        outputs = activation_function(Ws_plus_b)
    print(outputs)
    return outputs
# 构建隐藏层，假设隐藏层有10个神经元
h1 = add_layer(xs, 1, 20, activation_function=tf.nn.relu)
# 构建输出层，假设输出层和输入层一样，有1个神经元
prediction = add_layer(h1, 20, 1, activation_function=None)
# 计算预测值与真实值间的误差
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),
                                     reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

# STEP-3 训练模型 TensorFlow训练1000次，每50次输出训练的损失值
# 初始化所有变量
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for i in range(10):
    sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
    if i % 5 == 0:
        print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))



