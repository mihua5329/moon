import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import re
import unicodedata
import numpy as np
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

device = torch.device("cpu")
use_cuda = False
# 数据文件
filename = "labeledTrainData.tsv"

# 创建文件并存储list对象
contentFile_All = 'contentFile_All1.dat'
f = open(contentFile_All, 'rb')
content_all = pickle.load(f)

document_info =  pd.read_csv(filename,sep='\t',encoding='UTF-8')
label = document_info['sentiment'].values.tolist()
review = content_all



# print('(********************************)')
class wordIndex(object):
    def __init__(self):
        self.count = 0
        self.word_to_idx = {}
        self.word_count = {}

    def add_word(self, word):
        if not word in self.word_to_idx:
            self.word_to_idx[word] = self.count
            self.word_count[word] = 1
            self.count += 1
        else:
            self.word_count[word] += 1

    def add_text(self, text):
        for i,word in enumerate(text):
            # print(word)
            self.add_word(word)
def limitDict(limit, classObj):
    dict1 = sorted(classObj.word_count.items(), key=lambda t: t[1], reverse=True)
    count = 0
    for x, y in dict1:
        if count >= limit - 1:
            classObj.word_to_idx[x] = limit
        else:
            classObj.word_to_idx[x] = count

        count += 1
vocabLimit = 20000
max_sequence_len = 500
obj1 = wordIndex()
for i,lines in enumerate(review):
    obj1.add_text(lines)
limitDict(vocabLimit, obj1)

df_train = pd.DataFrame({'review': review, 'label': label})

# 模型选择
x_train, x_test, y_train, y_test = train_test_split(df_train['review'].values,
                                                    df_train['label'].values, train_size=15000,
                                                    test_size=10000, shuffle=False, random_state=1)


class Model(torch.nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super(Model, self).__init__()
        self.hidden_dim = hidden_dim
        self.embeddings = nn.Embedding(vocabLimit + 1, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.linearOut = nn.Linear(hidden_dim, 2)

    def forward(self, inputs, hidden):
        x = self.embeddings(inputs).view(len(inputs), 1, -1)
        lstm_out, lstm_h = self.lstm(x, hidden)
        x = lstm_out[-1]
        x = self.linearOut(x)
        x = F.softmax(x)
        return x, lstm_h

    def init_hidden(self):
        if use_cuda:
            return (
            Variable(torch.zeros(1, 1, self.hidden_dim)).cuda(), Variable(torch.zeros(1, 1, self.hidden_dim)).cuda())
        else:
            return (Variable(torch.zeros(1, 1, self.hidden_dim)), Variable(torch.zeros(1, 1, self.hidden_dim)))
if use_cuda:
    model = Model(50, 100).cuda()
else:
    model = Model(50, 100)

loss_function = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
torch.save(model.state_dict(), 'model____2' + str(0) + '.pth')

for epoch in range(1):
    avg_loss = 0.0
    for i in range(15000):
        # 第一步: 请记住Pytorch会累加梯度.
        # 我们需要在训练每个实例前清空梯度
        model.zero_grad()
        # 此外还需要清空 LSTM 的隐状态,
        # 将其从上个实例的历史中分离出来.
        hidden = model.init_hidden()
        # 准备网络输入, 将其变为词索引的 Tensor 类型数据
        input_data = [obj1.word_to_idx[word] for word in x_train[i]]
        if len(input_data) > max_sequence_len:
            input_data = input_data[0:max_sequence_len]
        if use_cuda:
            input_data = Variable(torch.cuda.LongTensor(input_data))
        else:
            input_data = Variable(torch.LongTensor(input_data))
        target = y_train[i]

        if use_cuda:
            target_data = Variable(torch.cuda.LongTensor([target]))
        else:
            target_data = Variable(torch.LongTensor([target]))
        # 第三步: 前向传播
        y_pred, _ = model(input_data, hidden)
        y_pred = torch.log(y_pred)
        # 第四步: 计算损失和梯度值, 通过调用 optimizer.step() 来更新梯度
        loss = loss_function(y_pred, target_data)
        avg_loss += loss.data

        if i % 1000 == 0 or i == 1:
            print('epoch :%d iterations :%d loss :%g' % (epoch, i, loss.data))
        loss.backward()
        optimizer.step()
    torch.save(model.state_dict(), 'model____2' + str(i + 1) + '.pth')
    print('the average loss after completion of %d epochs is %g' % ((i + 1), (avg_loss / 15000)))

y = np.zeros((15000,1))
a = 0
list_final = []
with torch.no_grad():
     for i in range(15000):
        input_data = [obj1.word_to_idx[word] for word in x_train[i]]
        if len(input_data) > max_sequence_len:
            input_data = input_data[0:max_sequence_len]

        if use_cuda:
            input_data = Variable(torch.cuda.LongTensor(input_data))
        else:
            input_data = Variable(torch.LongTensor(input_data))
        hidden = model.init_hidden()
        y_pred, _ = model(input_data, hidden)
        # print(y_pred,y_test[i])
        if(y_pred[0][0] < y_pred[0][1]):
            y[i][0] = 1
        list_final.append(y_pred[0][1])
print('召回率为：',recall_score(y_train,y))
print('F1：',f1_score(y_train,y))
print('accuracy_score：',accuracy_score(y_train,y))
list_orig=[]
for i in range(0, 15000, 1):
    list_orig.append(y_train[i])
fpr, tpr, threshold = roc_curve(list_orig, list_final)
roc_auc = auc(fpr, tpr)  ###计算auc的值

lw = 2
plt.figure(figsize=(8, 5))
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.4f)' % roc_auc)  ###假正率为横坐标，真正率为纵坐标做曲线
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()

y1 = np.zeros((10000,1))
a = 0
list_final = []
with torch.no_grad():
    for i in range(10000):
        input_data = [obj1.word_to_idx[word] for word in x_test[i]]
        if len(input_data) > max_sequence_len:
            input_data = input_data[0:max_sequence_len]

        if use_cuda:
            input_data = Variable(torch.cuda.LongTensor(input_data))
        else:
            input_data = Variable(torch.LongTensor(input_data))
        hidden = model.init_hidden()
        y_pred, _ = model(input_data, hidden)
        # print(y_pred,y_test[i])
        if (y_pred[0][0] < y_pred[0][1]):
            y1[i][0] = 1
        list_final.append(y_pred[0][1])
print('召回率为：', recall_score(y_test, y1))
print('F1：', f1_score(y_test, y1))
print('accuracy_score：', accuracy_score(y_test, y1))
list_orig = []
for i in range(0, 10000, 1):
    list_orig.append(y_test[i])
fpr, tpr, threshold = roc_curve(list_orig, list_final)
roc_auc = auc(fpr, tpr)  ###计算auc的值

lw = 2
plt.figure(figsize=(8, 5))
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.4f)' % roc_auc)  ###假正率为横坐标，真正率为纵坐标做曲线
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()