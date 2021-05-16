#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split # 载入数据分割函数train_test_split
import matplotlib.pyplot as plt
import os
isTrain=True
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
def show_result(image, points):
    plt.imshow(image, cmap='gray')
    for i in range(15):
        plt.plot(points[2*i], points[2*i + 1], 'ro')
    plt.show()

def export(pred_points, filename):
    submission_data = pd.DataFrame(pred_points)
    submission_data.to_csv(filename, index=False)

Train_Dir = 'D:/大数据资料/大三上课件/机器学习/期末大实验/program/base/data/train.csv'
Test_Dir = 'D:/大数据资料/大三上课件/机器学习/期末大实验/program/base/data/test.csv'
train_data = pd.read_csv(Train_Dir)
test_data = pd.read_csv(Test_Dir)

# use the previous value to fill the missing value
train_data.fillna(method='ffill', inplace=True)

# preparing training data
imga = []
for i in range(len(train_data)):
    img = train_data['Image'][i].split(' ')
    img = ['0' if x == '' else x for x in img]
    imga.append(img)

image_list = np.array(imga, dtype='float')
X_train = image_list.reshape(-1, 96, 96, 1)

# preparing training label
training = train_data.drop('Image', axis=1)
y_train = []
for i in range(len(train_data)):
    y = training.iloc[i, 1:]
    y_train.append(y)
y_train = np.array(y_train, dtype='float')

# preparing test data
timga = []
for i in range(len(test_data)):
    timg = test_data['Image'][i].split(' ')
    timg = ['0' if x == '' else x for x in timg]
    timga.append(timg)
timage_list = np.array(timga, dtype='float')
X_test = timage_list.reshape(-1, 96, 96, 1)
X_5=X_train.reshape(-1,96*96)
X_6=X_test.reshape(-1,96*96)
#分割测试集和验证集
X_t,X_v,y_t,y_v=train_test_split(X_5,y_train)

#模型的训练和保存
import tensorflow.compat.v1 as tf 
tf.disable_v2_behavior() 
import matplotlib.pyplot as plt


sess = tf.compat.v1.InteractiveSession()
#构建一个卷积神经网络
#所以placeholder()函数是在神经网络构建graph的时候在模型中的占位，此时并没有把要输入的数据传入模型，它只会分配必
#要的内存。等建立session，在会话中，运行模型的时候通过feed_dict()函数向占位符喂入数据。
x = tf.placeholder("float", shape=[None, 9216],name="x")
y_ = tf.placeholder("float", shape=[None, 30],name="y_")

def calculate_mse(predict, label):
    #mse_array = tf.reduce_mean((predict - label)**2, 0)
    mse_array = tf.reduce_mean((predict - label)**2)
    return tf.sqrt(mse_array)

# 定义变量初始化
def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)
# 定义卷积和池化操作
def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1], padding='SAME')
def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
# 第一层卷积
W_conv1 = weight_variable([3,3,1,32])
b_conv1 = bias_variable([32])
x_images = tf.reshape(x,[-1,96,96,1])# -1 根据输入的实际情况判断，比如50张图片就是 50*784=>50,28,28,1

h_conv1 = tf.nn.relu(conv2d(x_images,W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# 第二层卷积
W_conv2 = weight_variable([2,2,32,64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)
#print(h_pool2.shape)
#第三次卷积
W_conv3 = weight_variable([2,2,64,128])
b_conv3 = bias_variable([128])

h_conv3 = tf.nn.relu(conv2d(h_pool2,W_conv3) + b_conv3)
h_pool3 = max_pool_2x2(h_conv3)
# 全连接层
W_fc1 = weight_variable([128*12*12,500])
b_fc1 = bias_variable([500])

h_pool3_flat = tf.reshape(h_pool3,[-1,12*12*128])
h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat,W_fc1) + b_fc1)

# dropout
keep_prob = tf.placeholder("float",name="keep_prob")
h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)
# 全连接层(输出层)
W_fc2 = weight_variable([500,30])
b_fc2 = bias_variable([30])

#保存模型要用name初始化
#y_conv = tf.matmul(h_fc1_drop,W_fc2) + b_fc2
#y_conv = tf.convert_to_tensor(tf.matmul(h_fc1_drop,W_fc2) + b_fc2,name="y_conv")
y_conv = tf.matmul(h_fc1_drop,W_fc2) + b_fc2

# 训练和评估模型
cross_entropy = calculate_mse(y_conv,y_)
# 交叉熵代价函数
# Adam的优点主要在于经过偏置校正后，每一次迭代学习率都有个确定范围，使得参数比较平稳。
# 相比于基础SGD算法，1.不容易陷于局部优点。2.速度更快
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

#correct_prediction = tf.equal(tf.argmax(y_conv,1),tf.argmax(y_,1))
#accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"))
mse=calculate_mse(y_conv,y_)



def next_batch(train,target,batch_size):
    length=len(train)
    index=[i for i in range(length)]
    np.random.shuffle(index)
    cnt=length/batch_size+1
    while cnt>0:
        batch_x=[]
        batch_y=[]
        try:
            for i in range(batch_size):
                batch_x.append(train[index[i]])
                batch_y.append(target[index[i]])
                index.remove(index[i])
        except:
             index=[i for i in range(length)]
             continue
        
        yield (batch_x,batch_y)
a=next_batch(X_t,y_t,50)
"""saver = tf.train.Saver(max_to_keep=4)
ckpt_file_path = "D:/models12/mnist"
path = os.path.dirname(os.path.abspath(ckpt_file_path))
if os.path.isdir(path) is False:
    os.makedirs(path)"""
sess.run(tf.global_variables_initializer())
for i in range(10001):
    #batch[0] = X_t.next_batch(5)
    #batch[1] = y_t.next_batch(5)
    batch = next(a)
    if i%100 == 0:
        train_accuracy = mse.eval(feed_dict={x:batch[0],y_:batch[1],keep_prob:1.0})
        #train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
        print("step %d,测试均方误差 %g"%(i,train_accuracy))
    """if i%1000==0:
        tf.train.Saver().save(sess,ckpt_file_path,write_meta_graph=True)"""
    #模型保存
    train_step.run(feed_dict={x:batch[0],y_:batch[1],keep_prob:0.5})
print("验证集的均方误差： %g"%mse.eval(feed_dict={x:X_v,y_:y_v,keep_prob:1.0}))

y_result = y_conv.eval(feed_dict={x:X_v,y_:y_v,keep_prob:1.0})

show_result(X_v[0].reshape(96,96), y_result[0])
show_result(X_v[0].reshape(96,96), y_v[0])

y_test = y_conv.eval(feed_dict={x:X_6,keep_prob:1.0})
#保存预测结果
export(y_test,'result_13.csv')

