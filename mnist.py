import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

INPUT_NODE = 784
OUTPUT_NODE = 10
LAYER1_NODE = 500
REGULARATION_RATE = 0.001
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
BATCH_SIZE = 100
TRAINING_STEPS = 30000
MOVING_AVERAGE_DECAY = 0.99

def inference(input_tensor,avg_class,weight1,biase1,weight2,biase2):
    if avg_class == None:
        input_tensor = tf.nn.relu(tf.matmul(input_tensor,weight1)+biase1)
        return tf.matmul(input_tensor,weight2)+biase2
    else:
        input_tensor = tf.nn.relu(tf.matmul(input_tensor,avg_class.average(weight1))+avg_class.average(biase1))
        return tf.matmul(input_tensor,avg_class.average(weight2))+avg_class.average(biase2)

def train(mnist):
    x = tf.placeholder(tf.float32,shape=(None,INPUT_NODE),name='x-input')
    y_ = tf.placeholder(tf.float32,shape=(None,OUTPUT_NODE),name='y-input')

    weight1 = tf.Variable(tf.truncated_normal([INPUT_NODE,LAYER1_NODE],stddev=0.1))
    biase1 = tf.Variable(tf.constant(0.1,shape = [LAYER1_NODE]))

    weight2 = tf.Variable(tf.truncated_normal([LAYER1_NODE,OUTPUT_NODE],stddev=0.1))
    biase2 = tf.Variable(tf.constant(0.1,shape = [OUTPUT_NODE]))
    y = inference(x,None,weight1,biase1,weight2,biase2)

    global_step = tf.Variable(0,trainable=False)

    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,global_step)
    variable_averages_op = variable_averages.apply(tf.trainable_variables())
    average_y = inference(x,variable_averages,weight1,biase1,weight2,biase2)

    # 定义损失函数
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,labels=tf.argmax(y_,1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    # 正则化损失函数
    regularizer = tf.contrib.layers.l2_regularizer(REGULARATION_RATE)
    regularation = regularizer(weight1)+regularizer(weight2)

    loss = cross_entropy_mean+regularation


    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE,global_step,mnist.train.num_examples/BATCH_SIZE,LEARNING_RATE_DECAY)
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step)

    with tf.control_dependencies([train_step,variable_averages_op]):
        train_op = tf.no_op(name='train')
    correct_prediction = tf.equal(tf.argmax(average_y,1),tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        validate_feed={
            x: mnist.validation.images,
            y_: mnist.validation.labels
        }
        test_feed ={x:mnist.test.images,y_:mnist.test.labels}

        for i in range(TRAINING_STEPS):
            if i%1000 == 0:
                validate_acc = sess.run(accuracy,feed_dict=validate_feed)
                print("After %d training step(s),validation accuracy""using average model is %g"%(i,validate_acc))

            xs,ys = mnist.train.next_batch(BATCH_SIZE)
            sess.run(train_op,feed_dict={x:xs,y_:ys})

        #训练结束后，在测试数据上见此模型的最终正确率
        test_acc = sess.run(accuracy,feed_dict=test_feed)
        print("After % training step(s),test accuracy using average""model is %g"%(TRAINING_STEPS,test_acc))

def main(argv = None):
    mnist = input_data.read_data_sets("E:/learn/master/tensorflow/MNIST_data/",one_hot=True)
    train(mnist)

if __name__=='__main__':
    tf.app.run()


