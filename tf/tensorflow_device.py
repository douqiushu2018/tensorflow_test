import tensorflow as tf

with tf.device('/cpu:0'):
    v1 = tf.constant(1)

with tf.device('/cpu:1'):
    v2 = tf.constant(2)

with tf.device('/gpu:0'):
    v3 = tf.constant(3)

with tf.device('/gpu:1'):
    v4 = tf.constant(4)

config = tf.ConfigProto
# 显示变量实际分配到哪个设备上
config.log_device_placement = True
# 当设备不存在时，允许自动分配到其他设备
config.allow_soft_placement = True
# 启动时分配多少显存
config.gpu_options.per_process_gpu_memory_fraction = 0.5

with tf.Session(config=config) as sess:
    print(sess.run(v4))
