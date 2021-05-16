# 用到的库
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingClassifier# GBDT
from sklearn.metrics  import roc_curve,auc 
from sklearn.model_selection import train_test_split # 载入数据分割函数train_test_split


df = pd.read_csv('D:/大数据资料/大作业/train.csv')

# 特征值处理
df=df.join(pd.get_dummies(df.Gender))
df=df.join(pd.get_dummies(df.Vehicle_Age))
df=df.join(pd.get_dummies(df.Vehicle_Damage))

df.loc[:,"Driving_License"] = df.loc[:,"Driving_License"].fillna(1)

df=df.dropna()

df.drop(["id","Gender"],axis=1,inplace=True)


df.drop(["Vintage","Vehicle_Age","Vehicle_Damage"],axis=1,inplace=True)



# 训练集测试集划分


df_train,df_test = train_test_split(df,test_size = 0.25,stratify=df['Response'])

# 训练集预处理
data = df_train.Annual_Premium
data_re = data.values.reshape((data.index.size, 1))
k = 10 # 设置离散之后的数据段为10
k_model = KMeans(n_clusters = k, n_jobs = 4)
result = k_model.fit_predict(data_re)
df_train['premium'] = result
df_train.groupby('premium').premium.count()

df_train.drop(["Annual_Premium"],axis=1,inplace=True)

y_train=np.array(df_train.Response)
df_train.drop(["Response"],axis=1,inplace=True)
X_train=np.array(df_train)


# 测试集预处理
data = df_test.Annual_Premium
data_re = data.values.reshape((data.index.size, 1))
result = k_model.predict(data_re)
df_test['premium'] = result
df_test.groupby('premium').premium.count()

df_test.drop(["Annual_Premium"],axis=1,inplace=True)

y_test=np.array(df_test.Response)
df_test.drop(["Response"],axis=1,inplace=True)
X_test=np.array(df_test)

# 模型构造



gbdt=GradientBoostingClassifier(learning_rate = 0.05,min_samples_leaf = 70,max_features = 12,subsample= 0.8,random_state = 10,max_depth=5,n_estimators=300,min_samples_split=1400)
y_score = gbdt.fit (X_train, y_train) .decision_function (X_test) 
fpr, tpr, threshold = roc_curve (y_test, y_score) ### to calculate the true and false positive rate
roc_auc = auc(fpr,tpr) ###计算auc的值
print(roc_auc)


plt.figure(figsize=(10,10))
lw=2 
plt.plot(fpr, tpr, color="k",lw=lw, label='ROC curve (area = %0.9f)' % roc_auc) ###假正率为横坐标，真正率为纵坐标做曲线
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()



# 预测
df1 = pd.read_csv('D:/大数据资料/大作业/test.csv')


# 数据预处理
df1=df1.join(pd.get_dummies(df1.Gender))
df1=df1.join(pd.get_dummies(df1.Vehicle_Age))
df1=df1.join(pd.get_dummies(df1.Vehicle_Damage))
df1.loc[:,"Driving_License"] = df1.loc[:,"Driving_License"].fillna(1)
df1=df1.dropna()


df1.drop(["Gender","Vintage","Vehicle_Age","Vehicle_Damage"],axis=1,inplace=True)


data = df1.Annual_Premium
data_re = data.values.reshape((data.index.size, 1))
result = k_model.predict(data_re)
df1['premium'] = result
df1.groupby('premium').premium.count()
df1.drop(["Annual_Premium"],axis=1,inplace=True)
# 保留id
id=np.array(df1.id)
df1.drop(["id"],axis=1,inplace=True)

data=np.array(df1)
y_score1=gbdt.fit (X_train, y_train) .predict_proba(data)


dataframe= pd.DataFrame({'id':id,'Response':y_score1[:,1]})



#将DataFrame存储为csv,index表示是否显示行名，default=True

dataframe.to_csv("D:/大数据资料/大作业/result_1.csv",index=False,sep=',')







