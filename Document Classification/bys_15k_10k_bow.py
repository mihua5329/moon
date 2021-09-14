import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
# 数据文件
filename = "labeledTrainData.tsv"

# 创建文件并存储list对象
contentFile_All = 'contentFile_All1.dat'
f = open(contentFile_All, 'rb')
content_all = pickle.load(f)

document_info =  pd.read_csv(filename,sep='\t',encoding='UTF-8')
label = document_info['sentiment'].values.tolist()
review = content_all

df_train = pd.DataFrame({'review': review, 'label': label})

# 模型选择
x_train, x_test, y_train, y_test = train_test_split(df_train['review'].values,
                                                    df_train['label'].values, train_size=15000,
                                                    test_size=10000, shuffle=False, random_state=1)

# 处理训练集特征
words = []
for line_index in range(len(x_train)):
    try:
        words.append(' '.join(x_train[line_index]))
    except:
        print(line_index, x_train[line_index])
# print(x_train[0])
# print(words[0])

# 特征提取
# vectorizer = TfidfVectorizer(analyzer='word', max_features=4000, lowercase= False)
# vectorizer.fit(words)
#bow
cv= CountVectorizer(max_features=20000,lowercase= False)
cv.fit(words)
# 生成分类器
classifier = MultinomialNB()
classifier.fit(cv.transform(words), y_train)

print('训练集上的预测结果：')
y_pred = classifier.predict(cv.transform(words))[0:15000]
orig_result=y_train
print('召回率为：',recall_score(y_train,y_pred))
print('F1：',f1_score(y_train,y_pred))
print('accuracy_score：',accuracy_score(y_train,y_pred))
# predict_proba不同于predict，它返回的预测值为，获得所有结果的概率。可以用来进行计算fpt,tpr
predict_result = classifier.predict_proba(cv.transform(words))

list_orig = []
for i in range(0, 15000, 1):
    list_orig.append(y_train[i])
list_final = []
for i in range(0, 15000, 1):
    list_final.append(predict_result[i][1])
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

# 处理测试机特征
test_words = []
for line_index in range(len(x_test)):
    try:
        test_words.append(' '.join(x_test[line_index]))
    except:
        print(line_index, x_test[line_index])
# classifier.fit(vectorizer.transform(test_words), y_test)
print('测试集上的预测结果：')
y_pred = classifier.predict(cv.transform(test_words))[0:10000]
# print(y_pred)
orig_result=y_test
print('召回率为：',recall_score(y_test,y_pred))
print('F1：',f1_score(y_test,y_pred))
print('accuracy_score：',accuracy_score(y_test,y_pred))
#classifier.predict(vectorizer.transform(test_words))[0:10000]
predict_result = classifier.predict_proba(cv.transform(test_words))
list_orig=[]
for i in range(0, 10000, 1):
    list_orig.append(orig_result[i])

list_final = []
for i in range(0, 10000, 1):
    list_final.append(predict_result[i][1])

list_id = []
for i in range(15001, 25001, 1):
    list_id.append(i)


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

