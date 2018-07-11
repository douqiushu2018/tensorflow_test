
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

# STEP-1 数据加载
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# STEP-2 定义回归模型
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.matmul(x, W) + b
# 定义损失函数和优化器
# 输入的真实值的占位符
y_ = tf.placeholder(tf.float32, [None, 10])
# 我们用tf.nn.softmax_cross_entropy_with_logits 来计算预测值y与真实值y_的差值，并取均值
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
# 采用SGD作为优化器
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# STEP-3 训练模型
# 这里采用InteractiveSession()来创建交互式上下文的TensorFlow会话
# 与常规会话不同的是，交互式会话会成为默认会话
# 方法tf.Tensor.eval 和 tf.Operation.run都可以使用该会话来运行操作
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
# Train
for _ in range(1000):
    batch_x, batch_y = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_x, y_: batch_y})

# STEP-4 评估模型
# 评估训练好的模型
# 计算预测值和真实值
correct_predicition = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
# 布尔型转化为浮点数，并取平均值，得到准确率
accuracy = tf.reduce_mean(tf.cast(correct_predicition, tf.float32))
# 计算模型在测试集上的准确率
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))






