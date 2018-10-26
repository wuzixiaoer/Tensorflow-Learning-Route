import tensorflow as tf
INPUT_NODE = 784
LAY1_NODE = 500
OUTPUT_NODE = 10
def get_weight_variable(shape,regularizer):
    weights = tf.get_variable("weights",shape=shape,initializer=tf.truncated_normal_initializer(stddev=0.1))
    if regularizer!=None:
        tf.add_to_collection("losses",regularizer(weights))
    return weights
def inference(input_sensor,regularizer):
    with tf.variable_scope("layer1"):
        weights1 = get_weight_variable([INPUT_NODE,LAY1_NODE],regularizer)
        biases1 = tf.Variable(tf.constant(0.0,shape=[LAY1_NODE]))
        layer1 = tf.nn.relu(tf.matmul(input_sensor,weights1)+biases1)
    with tf.variable_scope("layer2"):
        weights2 = get_weight_variable([LAY1_NODE,OUTPUT_NODE],regularizer)
        biases2 = tf.Variable(tf.constant(0.0,shape=[OUTPUT_NODE]))
        layer2 = tf.matmul(layer1,weights2)+biases2
    return layer2
