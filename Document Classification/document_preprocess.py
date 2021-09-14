import pandas as pd
import re
import nltk
from nltk import stem
import pickle

# 数据文件
filename = "labeledTrainData.tsv"
# 停用词文件
stopfile = "stop_words.txt"
stemmer = stem.PorterStemmer()    # nltk.stem词干提取对象初始化
# 文本预处理1，字符串文本, 返回list格式的单词
def text_pre_process1(text):
    text_1 = re.sub(r"</?[^>]*>|\\|n*'[\w]*|[^(\w|\s)]", ' ', text)    # 去除html标签，'\', 英文缩写, 非英文字符
    text_2 = nltk.word_tokenize(text_1)    # nltk分词
    text_3 = []
    for word in text_2:
        text_3.append(stemmer.stem(word))    # nltk.stem词干提取
    return text_3
# 文本处理2，处理停用词，单词list，停用词list
def stop_words_process2(word_list, stop_list):
    word_clean = []
    for word in word_list:
        if word.lower() in stop_list:
            continue
        word_clean.append(word)
    return word_clean

#lookuperror
# nltk.download('stopwords')
# nltk.download('punkt')

# 获取停用词，文件为stopfile
stop_words_t = pd.read_csv(stopfile, index_col=False, sep='\t', names=['stopword'], quoting=3, encoding='utf-8')
stop_words = []
for line in stop_words_t['stopword']:
    stop_words.append(stemmer.stem(line))
# print(stop_words)

# 读取数据文件
document_info = pd.read_csv(filename,sep='\t',encoding='UTF-8')
content = document_info['review'].values.tolist()

# print(content[0])
# print(len(content[0]))
# print('******************')

content_list = []
for line in content:
    content_s1 = text_pre_process1(line)
    content_s2 = stop_words_process2(content_s1, stop_words)
    content_list.append(content_s2)


# print(content_list[0])
# print(content_list[1])
# print(content_list[2])
# print(content_list[3])
# print(content_list[4])
# print(len(content_list[0]), len(content_list[1]), len(content_list[2]), len(content_list[3]), len(content_list[3]))


# 创建文件并存储list对象
contentFile_All = 'contentFile_All1.dat'
f = open(contentFile_All, 'wb')
pickle.dump(content_list, f)
f.close()
# 删除list对象
# del content_list

# 取出并展示list对象
# print('***********2222***********')
#f = open(contentFile_All, 'rb')
#content_all = pickle.load(f)
#print(content_all)
#print(len(content_all))
