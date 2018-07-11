import tensorflow as tf

# 将输入的文件名列表产生一个FIFO的文件名队列
# shuffle=True 表示是否将文件顺序随机打乱
# num_epochs=2 表示读取数据最大的迭代数
filename_queue = tf.train.string_input_producer(['file0.csv',
                                                 'file1.csv'], shuffle=True, num_epochs=2)
# 采用读文本的reader，按文本的行读
reader = tf.TextLineReader()
key, value = reader.read(filename_queue)

# 默认值1.0，默认了输入数据的类型是float
record_defaluts = [[1.0], [1.0]]
# 采用decode_csv的方式来读取数据
# 默认是按逗号来分割每一行的数据
v1, v2 = tf.decode_csv(value, record_defaluts)
print(v1.dtype)
# 计算乘法
v_mul = tf.multiply(v1, v2)

init_op = tf.global_variables_initializer()
local_init_op = tf.local_variables_initializer()

sess = tf.Session()

sess.run(init_op)
sess.run(local_init_op)

# 启动输入数据的队列，否则会处于一直等待数据的状态
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)
try:
    while not coord.should_stop():
        value1, value2, mul_result = sess.run([v1, v2, v_mul])
        print('%f\t%f\t%f' %(value1, value2, mul_result))
except tf.errors.OutOfRangeError:
    print('Done...')
finally:
    coord.request_stop()

coord.join(threads)
sess.close()




