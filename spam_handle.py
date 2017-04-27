#coding:utf-8
import jieba
import pandas as pd
from gensim.models import word2vec
import numpy as np
from sklearn import svm
from sklearn.metrics import classification_report
from numpy import *

try:
   import cPickle as pickle
except:
   import pickle
import svm_xbb

def data_segment(content_data):
    count = 0
    cutword_file = open("../data/cut_result.txt", 'w')
    train_length = content_data.values.shape[0]*0.8
    for i in content_data.values[:train_length]:
        count += 1
        seg_list = jieba.cut(i[0], cut_all=False)
        for j in seg_list:
            cutword_file.write(j.encode("utf-8") + ' '.encode("utf-8"))
        if (count % 1000 == 0):
            print count
    cutword_file.close()

def word2vec_model(content_data,label_data):
    sentences = word2vec.Text8Corpus("../data/cut_result.txt")
    model = word2vec.Word2Vec(sentences, size=50, min_count=5)
    model.save_word2vec_format('../data/spam_word2vec_model',binary=True)
    #把word2vec模型存在本地

    train_X = []
    train_length = content_data.values.shape[0] * 0.8
    train_Y = []
    flag = 0
    for i in content_data.values[:train_length]:
        seg_list = jieba.cut(i[0], cut_all=False)
        count = 0
        for j in seg_list:
            if j not in model:
                continue
            if count == 0:
                old = model[j]
                new = np.zeros(shape=old.shape)
            else:
                new = model[j]
            old = old + new
            count += 1
        if (count != 0):
            old = old * (1.0 / count)
            train_X.append(list(old))
            train_Y.append(label_data[flag])
        flag += 1

    test_X = []
    test_Y = []
    for i in content_data.values[train_length:]:
        seg_list = jieba.cut(i[0], cut_all=False)
        count = 0
        for j in seg_list:
            if j not in model:
                continue
            if count == 0:
                old = model[j]
                new = np.zeros(shape=old.shape)
            else:
                new = model[j]
            old = old + new
            count += 1
        if (count != 0):
            old = old * (1.0 / count)
            test_X.append(list(old))
            test_Y.append(label_data[flag])
        else:
            test_X.append([0.0 for k in range(0,250)])
            test_Y.append(label_data[flag])
        flag += 1

    return train_X,train_Y,test_X,test_Y

def train(X_train,X_test,Y_train):
    clf = svm.SVC()
    clf.fit(X_train,Y_train)
    Y_predict = clf.predict(X_test)
    return Y_predict


def report(y_true, y_pred):
    print "classification_report(left: labels):"
    print classification_report(y_true, y_pred)

def model_save(alphs_result, x_result, y_result, b ):

    svm_model = open("../data/svm_model_1.txt", 'w')
    alphs_result_string = pickle.dumps(alphs_result)
    x_result_string = pickle.dumps(x_result)
    y_result_string = pickle.dumps(y_result)
    b_string = pickle.dumps(b)
    svm_model.write(alphs_result_string + "****&&" + x_result_string + "****&&" + y_result_string + "****&&" + b_string)
    svm_model.close()

if __name__ =="__main__":

    label_data = pd.read_table('../data/labeled.txt')
    content_data = label_data[["Content"]]
    label_data = label_data[["Label"]]
    label_data = label_data.values
    label_data.shape = (1,len(label_data))
    label_data = label_data[0]
    for i in range(len(label_data)):
        if(label_data[i] == 0):
            label_data[i]= -1
    X_train,Y_train,X_test,Y_test = word2vec_model(content_data,label_data)

    #利用sklearn的计算结果
    Y_predict = train(X_train,X_test,Y_train)
    report(Y_test, Y_predict)

    #测试自己实现的svm
    C = 0.6
    toler = 0.001
    maxIter = 100
    test = svm_xbb.SVM(X_train, Y_train, C, toler, 1, maxIter);
    print "step 1: training..."
    alphs_result, x_result, y_result, b = test.train_svm()
    model_save(alphs_result, x_result, y_result, b )
    print "step 2: testing..."
    Y_predict = svm_xbb.label(X_test, alphs_result, x_result, y_result, b)
    print "step 3: show the result..."
    report(Y_test, Y_predict)


