# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 19:20:19 2017

@author: osm
"""
from functools import reduce
import numpy
import gc
from keras.models import Sequential
from keras.layers import Dense, Flatten, Activation
from keras.layers import Dropout
from keras.layers.convolutional import Conv1D, Conv2D, MaxPooling1D, MaxPooling2D
from keras.utils import np_utils
from keras.optimizers import SGD, Adam
from keras.utils import plot_model
import pandas as pd
import csv
import json
import gc
import re
from keras import initializers
alphabet = set('0123456789abcdefghijklmnopqrstuvwxyz авбгдеёжзийклмнопрстуфхцчшщъыьэюя')
alphabet_dict = dict(zip(alphabet,list(range(len(alphabet)))))
#alphabet = set('ab')
#alphabet_dict = dict(zip(alphabet,list(range(len(alphabet)))))

max_chars = 100

#функция которая в строке разделяет все буквенно цифровые обозначения
def split_name(name):
    #сначала замению все разделяющие символы на пробел
    name=name.lower()
    name=re.sub(r"[-;,\./:\|\\\)\(\+\*\?\[\]\'\n\r\t\"\^\$\#\x01\x13\x02`~\x7f¦©«¬\xad®°±µ¶·»ђєіљћ–—’‚•…‰№™\&\<\=\>\@\_\{\}\‘\!\%“”„]",' ',name)
    #теперь делаю разделение буквенно-цифровых обозначений
    q = re.findall(r'[^\s ^\d]\d',name)+re.findall(r'\d[^\d ^\s]',name)
    #print(q)
    for elem in q:
        name = re.sub(elem,elem[0]+' '+elem[1],name)
    #сейчас сжимаю пробелы
    name = name.strip()
    name=re.sub('\s+',' ',name)
    return name

#загружаем данные
data = pd.read_csv('export.csv', sep= ',', encoding = 'cp1251', engine = 'python')
#разбиваем буквенно-цифровые обозначения
data=data.dropna()
data['NAME']=data['NAME'].apply(split_name)
#set(' '.join(data['NAME']))-alphabet
#перенумеровываем классы от 1 до K
uniq_cls=data['CLS_ID'].unique()
len(uniq_cls)
max_cls = len(uniq_cls)
cls_ser = pd.Series([i for i in range(len(uniq_cls))],index=uniq_cls)
data['NEW_CLS_ID'] = data['CLS_ID'].map(cls_ser)
data=data[['NEW_CLS_ID','NAME','PRJ_ID','PR']]
#делим данные на обучающую и тестовые выборки
data_selected = data[data['PRJ_ID']!=48].dropna()
test_data = data[(data['PRJ_ID']==48) & (data['PR']==0)].dropna()
del data
test_data = test_data[test_data['NEW_CLS_ID'].isin(data_selected['NEW_CLS_ID'])]

data_selected = data_selected.sample(frac=1).reset_index(drop=True)
data_selected.to_csv('data_selected.csv',index=False,encoding='utf-8')
test_data.to_csv('test_data.csv',index=False,encoding='utf-8')
data_selected=pd.read_csv('data_selected.csv',encoding='utf-8')
test_data=pd.read_csv('test_data.csv',encoding='utf-8')
uniq_cls=data_selected['NEW_CLS_ID'].unique()
len(uniq_cls)
max_cls=max(uniq_cls)+1

# Создаем последовательную модель
model = Sequential()
# Первый сверточный слой
model.add(Conv1D(256, (7), padding='same',
                        input_shape=(max_chars, len(alphabet)), activation='relu',use_bias=True,
                        kernel_initializer=initializers.random_normal(mean=0,stddev=0.02)))
model.add(MaxPooling1D(pool_size=(3)))
model.add(Conv1D(256, (7), activation='relu', padding='same',
                 kernel_initializer=initializers.random_normal(mean=0,stddev=0.02)))
model.add(MaxPooling1D(pool_size=(3)))
model.add(Conv1D(256, (3), activation='relu', padding='same',
                 kernel_initializer=initializers.random_normal(mean=0,stddev=0.02)))
model.add(Conv1D(256, (3), activation='relu', padding='same',
                 kernel_initializer=initializers.random_normal(mean=0,stddev=0.02)))
model.add(Conv1D(256, (3), activation='relu', padding='same',
                 kernel_initializer=initializers.random_normal(mean=0,stddev=0.02)))
model.add(Conv1D(256, (3), activation='relu', padding='same',
                 kernel_initializer=initializers.random_normal(mean=0,stddev=0.02)))
model.add(MaxPooling1D(pool_size=(3)))
model.add(Flatten())
model.add(Dense(1024, activation='relu',
                kernel_initializer=initializers.random_normal(mean=0,stddev=0.02)))
model.add(Dropout(0.5))
model.add(Dense(1024, activation='relu',
                kernel_initializer=initializers.random_normal(mean=0,stddev=0.02)))
model.add(Dropout(0.5))
model.add(Dense(max_cls, activation='softmax',
                kernel_initializer=initializers.random_normal(mean=0,stddev=0.02)))
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
adam = Adam()
model.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])
def get_phrase_presentation(str):
    res = numpy.zeros((max_chars,len(alphabet)))
    str=list(str[:max_chars])
    def tst(i):
        res[i][alphabet_dict[str[i]]] = 1
        return alphabet_dict[str[i]]
    list(map(tst,list(range(0,len(str)))))
    return res
def get_data_for_nn(block_size,data):
    while True:
        #print("while loop")
        x=numpy.zeros((block_size,max_chars,len(alphabet)))
        y=numpy.zeros((block_size))
        index = 0
        step = 1
        for review in data.itertuples():
            x[index-block_size*(step-1)] = get_phrase_presentation(review[2])
            y[index-block_size*(step-1)] = review[1]
            if index == (block_size*step - 1):
                step = step + 1
                #print("index: ", index)
                yield x,np_utils.to_categorical(y, max_cls)
                x = numpy.zeros((block_size,max_chars,len(alphabet)))
                y = numpy.zeros((block_size))
            index = index + 1
        if (len(data) % block_size) != 0:     
            x=numpy.resize(x,(len(data)-block_size*step,max_chars,len(alphabet)))
            y=numpy.resize(y,(len(data)-block_size*step))
            #print("epoch end: ", index)
            yield x,np_utils.to_categorical(y, max_cls)

test_lines = 0
max_lines = len(data_selected) - test_lines
block_size = 1000
steps_per_epoch = max_lines // block_size
if max_lines % block_size != 0:
    steps_per_epoch = steps_per_epoch + 1
model.fit_generator(get_data_for_nn(block_size=block_size,data=data_selected[:max_lines]),
                    steps_per_epoch = steps_per_epoch,
                    epochs = 3,
                    verbose = 1)

model.evaluate_generator(get_data_for_nn(block_size=block_size,data=test_data),
                         steps=len(test_data)//block_size,
                         workers=1)
scores = model.evaluate(X_test, Y_test, verbose=1)
scores = model.evaluate_generator(get_data_for_nn(block_size=block_size,
                                                  data=test_data),
                                                  steps=len(test_data)//block_size)
print("Точность работы на тестовых данных: %.2f%%" % (scores[1]*100))

plot_model(model,to_file='model.png',show_shapes=True, show_layer_names=False)
model.add(Conv2D(2, (3, 3), activation='relu', padding='same'))
plot_model(model,to_file='model.png',show_shapes=True, show_layer_names=False)

model.summary()
import pydot
print(pydot.find_graphviz())

from keras.models import Sequential
from keras.layers.convolutional import Conv2D
model = Sequential()
model.add(Conv2D(1, (1, 1), padding='same',
                        input_shape=(10, 10, 1), activation='relu'))
model.summary()

import matplotlib.pyplot as plt
data['NAME'].apply(len)
plt.hist(data['NAME'].apply(len))

scores = model.evaluate_generator(get_data_for_nn(block_size=block_size,data=data_selected[max_lines:]),steps=200)

print("Точность работы на тестовых данных: %.2f%%" % (scores[1]*100))
Точность работы на тестовых данных: 76.50%