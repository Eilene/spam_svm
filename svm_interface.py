#coding:utf-8

#输入一个string 首先分词，然后读取本地word2vec模型，计算向量,
#读取本地的svm模型，做预测
#返回预测结果
import warnings
warnings.filterwarnings("ignore")

import jieba
import numpy as np
from numpy import *
try:
   import cPickle as pickle
except:
   import pickle
import gensim

class svm_inter:
    def __init__(self,sentence):
        self.sentence = sentence

    def feature(self):
        word_model = gensim.models.Word2Vec.load_word2vec_format('../data/spam_word2vec_model',binary=True)
        seg_list = jieba.cut(self.sentence, cut_all=False)
        count = 0
        for j in seg_list:
            if j not in word_model:
                continue
            if count == 0:
                old = word_model[j]
                new = np.zeros(shape=old.shape)
            else:
                new = word_model[j]
            old = old + new
            count += 1
        if (count != 0):
            old = old * (1.0 / count)
            self.x = list(old)

        else:
            self.x = [0]*50


    def kernel(self,x, y, sigma):
        x = mat(x)
        y = mat(y)
        temp = x - y
        return math.exp(temp * temp.T / (-2) * sigma * sigma)

    def label(self, alphs_result, x_result, y_result, b):
        num = len(alphs_result)
        re = 0.0
        for i in range(num):
            re += alphs_result[i] * y_result[i] * self.kernel(x_result[i], self.x, 1)
        re += b
        if (re < 0):
            return -1
        else:
            return 1

    def get_label(self):
        read_model = open("../data/svm_mode_1.txt",'r')
        data_string = read_model.read().split("****&&")
        alphs_result = pickle.loads(data_string[0])
        x_result = pickle.loads(data_string[1])
        y_result = pickle.loads(data_string[2])
        b = pickle.loads(data_string[3])
        read_model.close()

        pre = self.label(alphs_result, x_result, y_result, b)
        if(pre == 1):
            print"垃圾短信"
        else:
            print "正常短信"
        return pre

sentence = "l l l"
test = svm_inter(sentence)
test.feature()
print sentence
pre =test.get_label()