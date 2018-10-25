import tensorflow as tf
from numpy.random import RandomState

batch_size=8
x = tf.placeholder(tf.float32,shape=(None,2),name='x')
y_ = tf.placeholder(tf.float32,shape=(None,1),name='y_')
w1 = tf.Variable(tf.random_normal([2,3],stddev=1,seed=1))
w2 = tf.Variable(tf.random_normal([3,1],stddev=1,seed=1))
# 前向传播
a = tf.matmul(x,w1)
y= tf.matmul(a,w2)
# 反向传播
cross_entropy = -tf.reduce_mean(y_*tf.log(tf.clip_by_value(y,1e-10,1.0)))

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(y,y_)

train_step = tf.train.AdamOptimizer().minimize(cross_entropy)
# 随机生成数据集
rdm = RandomState(1)
dataset_size =128
X = rdm.rand(dataset_size,2)
Y = [[int(x1+x2<1)] for (x1,x2) in X]
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    STEPS=5000
    for i in range(STEPS):
        start= (i*batch_size)%dataset_size
        end = min(start+batch_size,dataset_size)
        sess.run(train_step,feed_dict={x:X[start:end],y_:Y[start:end]})
        if i%1000==0:
            total_cross_entropy=sess.run(cross_entropy,feed_dict={x:X,y_:Y})
            print("After %d training step(s),cross entropy on all data is %g"%(i,total_cross_entropy))


