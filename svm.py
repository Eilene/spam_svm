#coding:utf-8
import numpy as np
import pandas as pd
from numpy import *
class SVM:
    def __init__(self,X_train,Y_train,c,termi_num,kernel_sigma,max_iternum):
        self.X_train = mat(X_train)
        self.Y_train = Y_train
        self.c = c
        self.termi_num = termi_num
        self.b = 0
        self.sample_length = len(X_train)
        self.alphs = np.zeros(shape = (self.sample_length,1))
        self.kernel_martix = self.get_kernel_martix(X_train,kernel_sigma)
        self.error = np.zeros(shape = (self.sample_length,1))
        # self.error = self.calcu_error()
        self.max_iternum = max_iternum


    def get_kernel_martix(self,X_train,kernel_sigma):
        kernel_temp_martix = np.zeros(shape = (self.sample_length,self.sample_length))
        if kernel_sigma == 0.0:
            kernel_sigma = 1.0
        for i in range(self.sample_length):
            for j in range(self.sample_length):

                temp = self.X_train[i]-self.X_train[j]
                kernel_temp_martix[i][j] = math.exp(temp*temp.T/(-2*kernel_sigma*kernel_sigma))
        return (kernel_temp_martix)

    # def calcu_error(self):
    #     for i in range(self.sample_length):
    #         temp_g = 0.0
    #         for j in range(self.sample_length):
    #             temp_g += self.alphs[j]*self.Y_train[j]*self.kernel_martix[i][j]
    #         temp_g = temp_g + self.b
    #         self.error[i] = temp_g - self.Y_train[i]
    #     return self.error

    # def update_error(self,i,j,b_old,old_alph1,old_alph2):
    #     for k in range(self.sample_length):
    #         self.error[k] = self.error[k]-b_old+self.b-old_alph1*self.Y_train[i]*self.kernel_martix[k][i]\
    #         -old_alph2*self.Y_train[j]*self.kernel_martix[k][j]+self.alphs[i]*self.Y_train[i]*self.kernel_martix[k][i]\
    #         +self.alphs[j]*self.Y_train[j]*self.kernel_martix[k][j]

    def calcu_one_error(self,i):
        temp_g = 0.0
        for j in range(self.sample_length):
            temp_g += self.alphs[j] * self.Y_train[j] * self.kernel_martix[i][j]
        temp_g = temp_g + self.b
        self.error[i] = temp_g - self.Y_train[i]

    def train_svm(self):
        #选择两个变量，利用smo优化，
        #第一个先从边界上选择违反kkt的，再从全部集合里面选违反kkt的
        #第二个选变化最大的
        #while
        iter_count = 0

        while(iter_count < self.max_iternum):
            pair_changed = 0
            iter_count += 1
            for i in range(self.sample_length):
                self.calcu_one_error(i)
                if (self.Y_train[i] * self.error[i] < -self.termi_num) and (self.alphs[i] <self.c) or \
                     (self.Y_train[i] * self.error[i] > self.termi_num) and (self.alphs[i] > 0):
                      # (math.fabs(self.Y_train[i] * self.error[i]) < self.termi_num) and (self.alphs[i] == 0 or self.alphs[i]==self.c)  :
                    #选择第二个变量
                    max_index = 0
                    max_value = 0.0
                    for j in range(self.sample_length):
                        if(i==j):
                            continue
                        self.calcu_one_error(j)
                        if math.fabs(self.error[i]-self.error[j])>max_value:
                            max_index = j
                            max_value = math.fabs(self.error[i]-self.error[j])
                    if(max_value==0.0):
                        continue

                    #i 和 max_index为选择的两个变量
                    if(self.Y_train[i] == self.Y_train[max_index]):
                        l = max(0,self.alphs[i]+self.alphs[max_index]-self.c)
                        h = min(self.c,self.alphs[i]+self.alphs[max_index])
                    else:
                        l = max(0,self.alphs[max_index]-self.alphs[i])
                        h = min(self.c,self.c+self.alphs[max_index]-self.alphs[i])
                    if(l==h):
                        continue

                    old_alph2 = self.alphs[max_index]
                    yita = self.kernel_martix[i][i] + self.kernel_martix[max_index][max_index] - 2*self.kernel_martix[i][max_index]
                    if(yita == 0):
                        continue
                    temp_alph2 = old_alph2 + (self.Y_train[max_index]*(self.error[i]-self.error[max_index])/(yita))
                    # temp_alph2 = temp_alph2[0]

                    if(temp_alph2>=h):
                        self.alphs[max_index] = h
                    elif(temp_alph2<=l):
                        self.alphs[max_index] = l
                    else:
                        self.alphs[max_index] = temp_alph2


                    if(math.fabs(old_alph2-self.alphs[max_index])<=self.termi_num):
                        continue

                    old_alph1 = self.alphs[i]
                    self.alphs[i] = old_alph1 + self.Y_train[i]*self.Y_train[max_index](self.alphs[max_index]-old_alph2)

                    b_old = self.b
                    b1_new = -self.error[i] - self.Y_train[i]*self.kernel_martix[i][i]*(self.alphs[i]-old_alph1)\
                             -self.Y_train[max_index]*self.kernel_martix[max_index][i]*(self.alphs[max_index]-old_alph2)+b_old

                    b2_new = -self.error[max_index] - self.Y_train[i] * self.kernel_martix[i][max_index] * (self.alphs[i] - old_alph1) \
                             - self.Y_train[max_index] * self.kernel_martix[max_index][max_index] * (self.alphs[max_index] - old_alph2) + b_old
                    self.b = (b1_new+b2_new)/2.0
                    # self.update_error(i, max_index, b_old, old_alph1, old_alph2)

                    pair_changed += 1

            if(pair_changed == 0):
                break

        alphs_result = []
        y_result = []
        x_result = []

        #模型训练完毕，计算分类面函数和w和b
        for i in range(self.sample_length):
            if(self.alphs[i]>0):
                #支持向量，
                alphs_result.append(self.alphs[i])
                y_result.append(self.Y_train[i])
                x_result.append(self.X_train[i])

        return alphs_result,x_result,y_result,self.b

def kernel(x,y,sigma):
    x = mat(x)
    y = mat(y)
    temp = x-y
    return math.exp(temp*temp.T/(-2)*sigma*sigma)

def label(x,alphs_result,x_result,y_result,b):
    pre = []
    for sample in x:
        num = len(alphs_result)
        re = 0.0
        for i in range(num):
            re += alphs_result[i]*y_result[i]*kernel(x_result[i],sample,1)
        re += b
        if(re<0):
            pre.append(-1)
        else:
            pre.append(1)
    return pre