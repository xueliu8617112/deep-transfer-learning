#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :dataprocessing.py
# @Time      :2021/3/2 10:45
# @Author    :Xue Liu

import numpy as np
import scipy.io as io
import os
import matplotlib.pyplot as plt
createVar = locals()

for i in range(1,11):
    path = r'E:\code-Xue\deep-transfer-learning\UDA\pytorch1.0\DSAN\data\accdsan%d.mat' %i
    f = io.loadmat(path)
    createVar['acc'+ str(i)] = f['acc']

acc = np.concatenate((acc1,acc2,acc3,acc4,acc5,acc6,
                      acc7,acc8,acc9,acc10,
                      # acc13,acc14,acc15,acc16,acc17,acc18,acc19,acc20
                      ),axis = 0)



for i in range(1,11):
    path = r'E:\code-Xue\deep-transfer-learning\UDA\pytorch1.0\DSAN\data\accdsdcl%d.mat' %i
    f = io.loadmat(path)
    createVar['bacc'+ str(i)] = f['acc']

acc1 = np.concatenate((bacc1,bacc2,bacc3,bacc4,bacc5,bacc6,
                      bacc7,bacc8,bacc9,bacc10,
                      # acc13,acc14,acc15,acc16,acc17,acc18,acc19,acc20
                      ),axis = 0)



plt.figure(1)
plt.boxplot(acc)
plt.boxplot(acc1)
plt.ylim(80, 90)
plt.show()


accavg = acc.mean(axis = 0)
accavg1 = acc1.mean(axis = 0)


plt.figure(2)
plt.plot(accavg)
plt.plot(accavg1)
plt.ylim(80, 90)
plt.show()

print(acc)