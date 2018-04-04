import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

#卷积网络
LOCAL_FOLDER = "E:/MNIST_data/"
mnist=input_data.read_data_sets(LOCAL_FOLDER,one_hot=True)

x= tf.placeholder("float",shape=[None,784])
y_=tf.placeholder("float",shape=[None,10])

x_image=tf.reshape(x,[-1,28,28,1])
#最后一维度是颜色通道数

print(x_image)

#随机值初始化权重
def weight_variable(shape):
    initial =tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)
#正值初始化bias
def bias_variable(shape):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)
#卷积 滑动窗口步长为1
def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')
#池化层 2*2
def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
#32个filters 每一个都有一个5*5的窗口 1是通道的数量
W_conv1=weight_variable([5,5,1,32])
#每一个权重均有一个bias 32个特征，针对每一个不同的特征均有一个bias
b_conv1=bias_variable([32])
#首先对x_image进行卷积，然后与bias求和 最后应用激活函数 将结果线性化
h_conv1=tf.nn.relu(conv2d(x_image,W_conv1)+b_conv1)
#进行池化
h_pool1=max_pool_2x2(h_conv1)

W_conv2=weight_variable([5,5,32,64])
b_conv2=bias_variable([64])

h_conv2=tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2)
h_pool2=max_pool_2x2(h_conv2)
#tensor的第一维度表示第二层卷积层的输出，大小为7*7带有64个filters，第二个参数是层中的神经元数量，我们可自由设置。
W_fc1=weight_variable([7*7*64,1024])
b_fc1=bias_variable([1024])
#接下来，我们将tensor打平到vector中。我们在之前章节中已经看到softmax需要将图片打平成一个vector任为输入。
# 通过打平后的vector乘以权重矩阵W_fc1，再加上bias b_fc1，最后应用ReLU激活函数后就能实现：
h_pool2_flat=tf.reshape(h_pool2,[-1,7*7*64])
h_fc1=tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1)
#保存神经元被保留的概率
keep_prob=tf.placeholder("float")
#通过dropout技术实现防止过拟合现象
h_fc1_drop=tf.nn.dropout(h_fc1,keep_prob)

W_fc2=weight_variable([1024,10])
b_fc2=bias_variable([10])
y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2)+b_fc2)



#通过交叉熵来评估模型好坏 其中y_conv是模型训练的结果 y_是期望的结果
cross_entropy=-tf.reduce_sum(y_*tf.log(y_conv))

train_step=tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction=tf.equal(tf.argmax(y_conv,1),tf.argmax(y_,1))
#将布尔值转换为float类型
accuracy=tf.reduce_mean(tf.cast(correct_prediction,"float"))

sess=tf.Session()

sess.run(tf.initialize_all_variables())
#现在我们已经准备好训练模型，模型会在卷积层中调整所有参数，一个全连接层来获得预测照片的结果。如果我们想知道模型表现如何，就需要之前的测试集。

#接下来的实现代码与上一章节中的十分类似，但有一个例外：将原来的梯度下降优化算法替换为ADAM优化算法，因为这个算法根据文献中的说明，它实现一种更有效的优化算法。

#同样还需要在feed_dict参数中提供keep_prob，用来控制dropout层保留神经元的概率。
for i in range(10):
    batch=mnist.train.next_batch(50)
    if i%10==0:
        train_accuracy=sess.run(accuracy,feed_dict={x:batch[0],y_:batch[1],keep_prob:1.0})

        print("step %d, training accuracy %g"%(i,train_accuracy))

    sess.run(train_step,feed_dict={x:batch[0],y_:batch[1],keep_prob:0.5})

print("test accuracy %g"%sess.run(accuracy,feed_dict={x:mnist.test.images,y_:mnist.test.labels,keep_prob:1.0}))