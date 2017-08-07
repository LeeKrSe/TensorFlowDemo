# -*- coding: utf-8 -*-

from PIL import Image
import tensorflow as tf
import numpy as np

w = 100
h = 100
c = 3

img_path = "../images/xinchang/"
model_dir = "./model/nobu/"
model_name = "nobunaga_model"

# -----------------构建网络----------------------
# 占位符
x = tf.placeholder(tf.float32, shape=[None, w, h, c], name='x')
y_ = tf.placeholder(tf.int32, shape=[None, ], name='y_')

# 第一个卷积层（100——>50)
conv1 = tf.layers.conv2d(
    inputs=x,
    filters=32,
    kernel_size=[5, 5],
    padding="same",
    activation=tf.nn.relu,
    kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

# 第二个卷积层(50->25)
conv2 = tf.layers.conv2d(
    inputs=pool1,
    filters=64,
    kernel_size=[5, 5],
    padding="same",
    activation=tf.nn.relu,
    kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

# 第三个卷积层(25->12)
conv3 = tf.layers.conv2d(
    inputs=pool2,
    filters=128,
    kernel_size=[3, 3],
    padding="same",
    activation=tf.nn.relu,
    kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)

# 第四个卷积层(12->6)
conv4 = tf.layers.conv2d(
    inputs=pool3,
    filters=128,
    kernel_size=[3, 3],
    padding="same",
    activation=tf.nn.relu,
    kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
pool4 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=2)

re1 = tf.reshape(pool4, [-1, 6 * 6 * 128])

# 全连接层
dense1 = tf.layers.dense(inputs=re1,
                         units=1024,
                         activation=tf.nn.relu,
                         kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                         kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))
dense2 = tf.layers.dense(inputs=dense1,
                         units=512,
                         activation=tf.nn.relu,
                         kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                         kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))
logits = tf.layers.dense(inputs=dense2,
                         units=5,
                         activation=None,
                         kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                         kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))
# ---------------------------网络结束---------------------------
loss = tf.losses.sparse_softmax_cross_entropy(labels=y_, logits=logits)
train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
correct_prediction = tf.equal(tf.cast(tf.argmax(logits, 1), tf.int32), y_)
acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

saver = tf.train.Saver()

# Load model
model_file=tf.train.latest_checkpoint(model_dir)
saver.restore(sess, model_file)

# 加载信长头像，正确的分类是0
imgs = []
labels = []

img = Image.open(img_path + "00034_00001.jpg")
img = img.resize((w, h))
img = np.array(img)
imgs.append(img)
labels.append(0)

imgs = np.asarray(imgs, np.float32)
labels = np.asarray(labels, np.float32)

print imgs.shape

ret = sess.run(y_, feed_dict={x: imgs, y_:labels})
print("计算模型结果成功！")
# 显示测试结果
print("预测结果:%d" % ret)
print("实际结果:%d" % 0)

# 加载信喵头像，正确的分类是1
imgs = []
labels = []

img = Image.open(img_path + "00034_01904.jpg")
img = img.resize((w, h))
img = np.array(img)
imgs.append(img)
labels.append(1)

imgs = np.asarray(imgs, np.float32)
labels = np.asarray(labels, np.float32)

# 根据模型计算结果
ret = sess.run(y_, feed_dict={x: imgs, y_:labels})
print("计算模型结果成功！")
# 显示测试结果
print("预测结果:%d" % ret)
print("实际结果:%d" % 1)
sess.close()

