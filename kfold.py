from utils import pickle_dump, pickle_load
from sklearn.model_selection import StratifiedKFold
from random import shuffle
import pandas as pd
import numpy as np
#210+75=285
training_file = 'training_ids'
validation_file = 'validation_ids'
training_list = list()
test_list = list()
label_list = pd.read_csv("./train_label.csv")
label_list = label_list['ret']
for i in range(label_list.__len__()):
    if int(label_list[i]) == 0:
        training_list.append(i)
    if int(label_list[i]) == 1:
        test_list.append(i)
shuffle(training_list)
shuffle(test_list)
print(training_list.__len__(),test_list.__len__())
pickle_dump(training_list, 'train_ids6.pkl')
'''
N = int(training_list.__len__() / 5)
print(training_list.__len__(),N)
j = 1
for i in range(0, training_list.__len__(), N):
    if (i+N) > training_list.__len__():
        continue
    train_file = training_file + str(j) + '.pkl'
    test_file = validation_file + str(j) + '.pkl'
    j = j+1
    if i==0:
        i=1
    #print(i, ' ', i+N-1)
    print(0, i-1, i, i+N-1, i+N, training_list.__len__()-1)
    test = training_list[i : i+N-1]
    train = training_list[0:i-1]+training_list[i+N:training_list.__len__()-1]
    pickle_dump(train, train_file)
    pickle_dump(test,  test_file)'''
