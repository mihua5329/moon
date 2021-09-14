## 数据集

labeledTrainData.tsv

## 预处理

document_preprocess.py 

输入：labeledTrainData.tsv、stop_words.txt

输出：contentFile_All1.dat

## 训练

### 贝叶斯——bow

python bys_15k_10k_bow.py

输入：labeledTrainData.tsv、contentFile_All1.dat

### 贝叶斯——tfidf

python bys_15k_10k_tfidf.py

输入：labeledTrainData.tsv、contentFile_All1.dat

### LSTM

python lstm_2

输入：labeledTrainData.tsv、contentFile_All1.dat

输出模型：model____115000.pth

python lstm_加dropout

输入：labeledTrainData.tsv、contentFile_All1.dat

输出模型：model__415000.pth

## 测试 LSTM

python model_predict.py

输入：model____115000.pth

