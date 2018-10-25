import tensorflow as tf

def get_weight(shape,lamda):
    var = tf.Variable(tf.random_normal(shape),dtype=tf.float32)
    tf.add_to_collection('losses',tf.contrib.layers.l2_regularizer(lamda)(var))
    return var

x = tf.placeholder(tf.float32,shape=(None,2))
y_ = tf.placeholder(tf.float32,shape=(None,1))

dimension_layer = [2,10,10,10,1]
layers = len(dimension_layer)
batch_size = 8

cur_layer = x
# 记录当前层
in_dimension = dimension_layer[0]

# 循环生成5层神经网络
for i in range(1,layers):
    out_dimension = dimension_layer[i]
    weight = get_weight([in_dimension,out_dimension],0.001)
    bias = tf.Variable(tf.constant(0.1,shape=[out_dimension]))
    cur_layer = tf.nn.relu(tf.matmul(cur_layer,weight)+bias)
    in_dimension = dimension_layer[i]

mse_loss = tf.reduce_mean(tf.square(y_-cur_layer))
tf.add_to_collection('losses',mse_loss)
loss = tf.add_n(tf.get_collection('losses'))

