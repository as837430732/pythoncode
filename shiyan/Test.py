import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
#神经网络
LOCAL_FOLDER = "E:/MNIST_data/"
np.random.seed(10)
tf.set_random_seed(10)

data = input_data.read_data_sets(LOCAL_FOLDER, one_hot=True)

def dense_layer(x, in_dim, out_dim, layer_name, act):
    with tf.name_scope(layer_name):
        # W
        weights = tf.Variable(
            tf.random_uniform(
                [in_dim, out_dim],
                maxval = tf.sqrt(6.0) / tf.sqrt(float(out_dim+in_dim)),
                seed = 10
            ), name="weights"
        )
        #tf.random_uniform(shape,minval=0,maxval=None,dtype=tf.float32,seed=None,name=None)
        #random_uniform:均匀分布随机数，范围为[minval,maxval]
        # b
        biases = tf.Variable(tf.zeros([out_dim]), name="biases")#初始化变量b为0

        # y =f(Wx+b)
        layer = act(tf.matmul(x, weights) + biases, name="activations")#通过激励函数

    return layer
input = tf.placeholder(tf.float32, [None, 784], name="input")
#tensorx将用来存储MNIST图像向量中784个浮点值（None代表该维度可为任意大小，本例子中将会是学习过程中照片的数量
targets = tf.placeholder(tf.float32, [None, 10], name="targets")
#期望的输出值

# 4、占位符
#
# 变量在定义时要初始化，但是如果有些变量刚开始我们并不知道它们的值，无法初始化，那怎么办呢？
#
# 那就用占位符来占个位置，如：
#
# x = tf.placeholder(tf.float32, [None, 784])
# 指定这个变量的类型和shape，以后再用feed的方式来输入值。

# network layers: two hidden and one output
hidden1 = dense_layer(input, 784, 200, "hidden1", act=tf.nn.relu)#relu激活函数
#hidden1 = dense_layer(input, 784, 200, "hidden1", act=tf.identity)
#hidden2 = dense_layer(hidden1, 200, 300, "hidden2", act=tf.nn.relu)
output = dense_layer(hidden1, 200, 10, "output", act=tf.identity)


# loss function: cross-entropy with built-in
# (stable) computation of softmax from logits

#交叉熵，将训练输出值与期望输出值进行比较   交叉熵的作用就是用来评估训练参数的好坏
# 首先看输入logits，它的shape是[batch_size, num_classes] ，一般来讲，就是神经网络最后一层的输入z。
#反向传播算法与梯度下降算法来最小化交叉熵
# 另外一个输入是labels，它的shape也是[batch_size, num_classes]，就是我们神经网络期望的输出。
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(#激活函数的交叉熵
        labels=targets, logits=output
    )
)


# training algorithm: Adam with configurable learning rate
#train_step = tf.train.AdamOptimizer(0.0001).minimize(cross_entropy)
#train_step = tf.train.MomentumOptimizer(0.01, 0.9).minimize(cross_entropy)
#train_step = tf.train.AdagradOptimizer(0.1).minimize(cross_entropy)
#train_step = tf.train.RMSPropOptimizer(0.1).minimize(cross_entropy)
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)
#因为我们使用了MNIST数据集，下面的代码显示我们使用，同时学习速率为0.1。
#train_step = tf.train.AdadeltaOptimizer(1.0).minimize(cross_entropy)

# evaluation operation: ratio of correct predictions
#tf.argmax(y,1)函数会返回tensor中参数指定的维度中的最大值的索引
#tf.argmax(output,1)是我们模型中输入数据的最大概率标签，tf.argmax(targets,1)是实际的标签
correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(targets, 1))
#将布尔值  转换为浮点值   例如，[True, False, True, True]会转换成[1,0,1,1]，其平均值0.75代表了准确比例。
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

from matplotlib import pyplot
import numpy as np

steps = []
accuracies = []

# creating session
sess = tf.InteractiveSession()


# initializing trainable variables
sess.run(tf.global_variables_initializer())

# training loop
for step in range(500):
    # fetching next batch of training data
    # 每次迭代中，从训练数据集中随机选取100张图片作为一批。
    batch_xs, batch_ys = data.train.next_batch(100)

    if step % 100 == 0:
        # reporting current accuracy of the model on every 100th batch
        #表示之前获得的输入分别赋给相关的placeholders
        batch_accuracy = sess.run(accuracy, feed_dict={input: batch_xs, targets: batch_ys})
        #print("{0}:\tbatch accuracy {1:.2f}".format(step, batch_accuracy))
        steps.append(step/100)
        accuracies.append(batch_accuracy)

    # running the training step with the fetched batch
    sess.run(train_step, feed_dict={input: batch_xs, targets: batch_ys})

pyplot.plot(steps, accuracies, 'k-', linewidth=2.0, color='green')
pyplot.show()


# evaluating model prediction accuracy of the model on the test set
#使用mnist.test数据集作为feed_dict参数来计算准确率
test_accuracy = sess.run(accuracy, feed_dict={input: data.test.images, targets: data.test.labels})


print("-------------------------------------------------")
print("Test set accuracy: {0:.4f}".format(test_accuracy))