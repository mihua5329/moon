#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
def export(pred_points, filename):
    submission_data = pd.DataFrame(pred_points)
    submission_data.to_csv(filename, index=False)
Test_Dir = './data/test.csv'
test_data = pd.read_csv(Test_Dir)
timga = []
for i in range(len(test_data)):
    timg = test_data['Image'][i].split(' ')
    timg = ['0' if x == '' else x for x in timg]
    timga.append(timg)
timage_list = np.array(timga, dtype='float')
X_test = timage_list.reshape(-1, 96, 96, 1)
X_6=X_test.reshape(-1,96*96)

import tensorflow.compat.v1 as tf 
tf.disable_v2_behavior() 
#加载模型进行预测
tf.reset_default_graph()
sess = tf.Session()
#加载模型
saver = tf.train.import_meta_graph('./models11/mnist.meta')
saver.restore(sess, tf.train.latest_checkpoint('./models11'))
 
 
graph = tf.get_default_graph()
x_ = graph.get_tensor_by_name("x:0")
keep_prob_ = graph.get_tensor_by_name("keep_prob:0")
yr = graph.get_tensor_by_name("y_conv:0")


y_test = sess.run(yr,feed_dict={x_:X_6,keep_prob_:1.0})
#保存预测结果
export(y_test,'result_11.csv')

