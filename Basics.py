import tensorflow as tf

'''
Some basic operations.
'''
node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0)

sess = tf.Session()
print(sess.run([node1, node2]))
sess.close()

a = tf.constant(5.0)
b = tf.constant(6.0)

c = a * b

d = tf.placeholder(tf.float32)
e = tf.placeholder(tf.float32)

addedNode = d + e

with tf.Session() as sess:
    fileWriter = tf.summary.FileWriter('graph', sess.graph)
    print(sess.run(c))
    print(sess.run(addedNode,{d:[1,3], e:[2,4]}))

'''
Define Model and calculate error
'''

# Model Parameters
var1 = tf.Variable([0.3], tf.float32)
var2 = tf.Variable([-0.3], tf.float32)

# Inputs and Outputs
inputVal = tf.placeholder(tf.float32)

linearModel = var1 * inputVal + var2  # actual output

y = tf.placeholder(tf.float32)  # expected output

# Loss
squaredDelta = tf.square(linearModel - y)
loss = tf.reduce_sum(squaredDelta)

# Optimize
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    # print(sess.run(loss, {inputVal: [1, 2, 3, 4], y: [0, -1, -2, -3]}))

    for i in range(1000):
        sess.run(train, {inputVal: [1, 2, 3, 4], y: [0, -1, -2, -3]})

    print(sess.run([var1, var2]))
