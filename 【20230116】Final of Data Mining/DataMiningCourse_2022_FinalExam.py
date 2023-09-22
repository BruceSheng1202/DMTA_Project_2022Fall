#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Mining Theory and Practice Course 2022 Final Exam

"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


# Pandas读取CSV格式数据

Training_Data = pd.read_csv('TrainingData.csv')
Training_Labels = pd.read_csv('TrainingLabels.csv')
Testing_Data = pd.read_csv('TestingData.csv')
Testing_Labels = pd.read_csv('TestingLabels.csv')


# 训练数据坐标
Training_Data = np.array(Training_Data.values, dtype=float)
Training_Data = Training_Data[:,1:3]

# 训练数据的类标签 (Class 1为阴性Negative, Class 2为阳性Positive)
Training_Labels = np.array(Training_Labels.values, dtype=int)
Training_Labels  = Training_Labels [:,1]

# 测试数据坐标
Testing_Data = np.array(Testing_Data.values, dtype=float)
Testing_Data = Testing_Data[:,1:3]

# 测试数据标签(Class 1为阴性Negative, Class 2为阳性Positive), 可用于计算混淆矩阵, 准确度, 敏感度, 特异度, 精确度
Testing_Labels = np.array(Testing_Labels.values, dtype=int)
Testing_Labels  = Testing_Labels [:,1]


# 训练数据作图, 未分类或聚类前的测试数据作图
# display scatters
plt.figure()
plt.xlabel("x axis")
plt.ylabel("y axis")
plt.scatter(Training_Data[Training_Labels == 1, 0:1], Training_Data[Training_Labels == 1, 1:2], marker='o', edgecolors='cornflowerblue', alpha=0.7, label='Class 1')
plt.scatter(Training_Data[Training_Labels == 2, 0:1], Training_Data[Training_Labels == 2, 1:2], marker='^', edgecolors='red', alpha=0.5, label='Class 2')
plt.scatter(Testing_Data[:,0:1], Testing_Data[:,1:2], marker='s', edgecolor='g', alpha=0.5, label='unknown')
plt.xlim(min(Training_Data[:,0])-1, max(Training_Data[:,0])+1)
plt.ylim(min(Training_Data[:,1])-1, max(Training_Data[:,1])+1)
plt.legend(loc='upper left')
plt.show()



# 测试数据最终结果作图(仅供参考)
# display scatters
plt.figure()
plt.xlabel("x axis")
plt.ylabel("y axis")
plt.scatter(Training_Data[Training_Labels == 1, 0:1], Training_Data[Training_Labels == 1, 1:2], marker='o', edgecolors='cornflowerblue', alpha=0.7, label='Class 1')
plt.scatter(Training_Data[Training_Labels == 2, 0:1], Training_Data[Training_Labels == 2, 1:2], marker='^', edgecolors='red', alpha=0.5, label='Class 2')
plt.scatter(Testing_Data[Testing_Labels == 1, 0:1], Testing_Data[Testing_Labels == 1, 1:2], marker='o', edgecolor='g', alpha=0.5, label='Testing Class 1')
plt.scatter(Testing_Data[Testing_Labels == 2, 0:1], Testing_Data[Testing_Labels == 2, 1:2], marker='^', edgecolor='g', alpha=0.5, label='Testing Class 2')
plt.xlim(min(Training_Data[:,0])-1, max(Training_Data[:,0])+1)
plt.ylim(min(Training_Data[:,1])-1, max(Training_Data[:,1])+1)
plt.legend(loc='upper left')
plt.show()