#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import random
import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from utils import loadWord2Vec, clean_str
from math import log
from sklearn import svm
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
import sys
from scipy.spatial.distance import cosine


# In[2]:

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--supportTxt', type=str, default='SizeAll', choices=['Size1', 'Size2', 'Size3', 'Size4', 'SizeAll'])
parser.add_argument('--contextDB', type=str, default='classification_gold_context', choices=['classification_gold_context','classification_1_context','classification_3_context','classification_5_context','classification_7_context','classification_9_context'])


# In[3]:
args = parser.parse_args()
Path='data/'+args.contextDB+'/'+args.supportTxt

# Path='data/classification_gold_context/SizeAll'

word_embeddings_dim = 300
word_vector_map = {}

# shulffing
doc_name_list = []
doc_dev_list = []
doc_train_list = []
doc_test_list = []

f = open(Path+'/all.txt', 'r')
lines = f.readlines()
for line in lines:
    doc_name_list.append(line.strip())
    temp = line.split("\t")
    if temp[1].find('test') != -1:
        doc_test_list.append(line.strip())
    elif temp[1].find('train') != -1:
        doc_train_list.append(line.strip())
    elif temp[1].find('val') != -1:
        doc_dev_list.append(line.strip())
f.close()


# In[3]:


doc_content_list = []
f = open(Path+'/corpus/all.clean.txt', 'r')
lines = f.readlines()
for line in lines:
    doc_content_list.append(line.strip())
f.close()
# print(doc_content_list)

dev_ids = []
for dev_name in doc_dev_list:
    dev_id = doc_name_list.index(dev_name)
    dev_ids.append(dev_id)
# print(dev_ids)
random.shuffle(dev_ids)

dev_ids_str = '\n'.join(str(index) for index in dev_ids)
f = open(Path+'/all.dev.index', 'w')
f.write(dev_ids_str)
f.close()

train_ids = []
for train_name in doc_train_list:
    train_id = doc_name_list.index(train_name)
    train_ids.append(train_id)
# print(train_ids)
random.shuffle(train_ids)

# partial labeled data
#train_ids = train_ids[:int(0.2 * len(train_ids))]

train_ids_str = '\n'.join(str(index) for index in train_ids)

train_ids_str=train_ids_str+'\n'+dev_ids_str
f = open(Path+'/all.train.index', 'w')
f.write(train_ids_str)
f.close()

test_ids = []
for test_name in doc_test_list:
    test_id = doc_name_list.index(test_name)
    test_ids.append(test_id)
# print(test_ids)
random.shuffle(test_ids)

test_ids_str = '\n'.join(str(index) for index in test_ids)
f = open(Path+'/all.test.index', 'w')
f.write(test_ids_str)
f.close()


ids = dev_ids + train_ids + test_ids
# print(ids)
# print(len(ids))

shuffle_doc_name_list = []
shuffle_doc_words_list = []
for id in ids:
    shuffle_doc_name_list.append(doc_name_list[int(id)])
    shuffle_doc_words_list.append(doc_content_list[int(id)])
    
shuffle_doc_name_str = '\n'.join(shuffle_doc_name_list)
shuffle_doc_words_str = '\n'.join(shuffle_doc_words_list)


# In[4]:


f = open(Path+'/all_shuffle.txt', 'w')
f.write(shuffle_doc_name_str)
f.close()

f = open(Path+'/corpus/all_shuffle.txt', 'w')
f.write(shuffle_doc_words_str)
f.close()

# build vocab
word_freq = {}
word_set = set()
for doc_words in shuffle_doc_words_list:
    words = doc_words.split()
    for word in words:
        word_set.add(word)
        if word in word_freq:
            word_freq[word] += 1
        else:
            word_freq[word] = 1

vocab = list(word_set)
vocab_size = len(vocab)

word_doc_list = {}

for i in range(len(shuffle_doc_words_list)):
    doc_words = shuffle_doc_words_list[i]
    words = doc_words.split()
    appeared = set()
    for word in words:
        if word in appeared:
            continue
        if word in word_doc_list:
            doc_list = word_doc_list[word]
            doc_list.append(i)
            word_doc_list[word] = doc_list
        else:
            word_doc_list[word] = [i]
        appeared.add(word)

word_doc_freq = {}
for word, doc_list in word_doc_list.items():
    word_doc_freq[word] = len(doc_list)

word_id_map = {}
for i in range(vocab_size):
    word_id_map[vocab[i]] = i

vocab_str = '\n'.join(vocab)

f = open(Path+'/corpus/all_vocab.txt', 'w')
f.write(vocab_str)
f.close()


# In[5]:


# label list
label_set = set()
for doc_meta in shuffle_doc_name_list:
    temp = doc_meta.split('\t')
    for lable in temp[2].split(' '):
        label_set.add(lable) 
label_list = list(label_set)
label_list
label_list_str = '\n'.join(label_list)
f = open(Path+'/corpus/all_labels.txt', 'w')
f.write(label_list_str)
f.close()


# In[6]:



# # x: feature vectors of training docs, no initial features
# # slect 90% training set
# train_size = len(train_ids)
# val_size = int(0.1 * train_size)
# real_train_size = train_size - val_size  # - int(0.5 * train_size)
# # different training rates

# real_train_doc_names = shuffle_doc_name_list[:real_train_size]
# real_train_doc_names_str = '\n'.join(real_train_doc_names)

real_train_doc_names = []
for train_name in doc_train_list:
    real_train_doc_names.append(train_name)
real_train_doc_names_str = '\n'.join(real_train_doc_names)

f = open(Path+'/all.real_train.name', 'w')
f.write(real_train_doc_names_str)
f.close()


# In[7]:



row_x = []
col_x = []
data_x = []
real_train_size= len(train_ids)
train_size=real_train_size + len(dev_ids)
for i in range(real_train_size):
    doc_vec = np.array([0.0 for k in range(word_embeddings_dim)])
    doc_words = shuffle_doc_words_list[i]
    words = doc_words.split()
    doc_len = len(words)
    for word in words:
        if word in word_vector_map:
            word_vector = word_vector_map[word]
            # print(doc_vec)
            # print(np.array(word_vector))
            doc_vec = doc_vec + np.array(word_vector)

    for j in range(word_embeddings_dim):
        row_x.append(i)
        col_x.append(j)
        # np.random.uniform(-0.25, 0.25)
        data_x.append(doc_vec[j] / 1 if doc_len==0 else doc_len)  # doc_vec[j]/ doc_len


# In[8]:


# x = sp.csr_matrix((real_train_size, word_embeddings_dim), dtype=np.float32)
x = sp.csr_matrix((data_x, (row_x, col_x)), shape=(
    real_train_size, word_embeddings_dim))


# In[9]:


### error 'extends differences' is not in list
y = []
for i in range(real_train_size):
    doc_meta = shuffle_doc_name_list[i]
    temp = doc_meta.split('\t')
    one_hot = [0 for l in range(len(label_list))]
    for label in temp[2].split(' '):
        label_index = label_list.index(label)
        one_hot[label_index] = 1
    y.append(one_hot)
y = np.array(y)
# print(y)


# In[10]:


# tx: feature vectors of test docs, no initial features
test_size = len(test_ids)

row_tx = []
col_tx = []
data_tx = []
for i in range(test_size):
    doc_vec = np.array([0.0 for k in range(word_embeddings_dim)])
    doc_words = shuffle_doc_words_list[i + train_size]
    words = doc_words.split()
    doc_len = len(words)
    for word in words:
        if word in word_vector_map:
            word_vector = word_vector_map[word]
            doc_vec = doc_vec + np.array(word_vector)

    for j in range(word_embeddings_dim):
        row_tx.append(i)
        col_tx.append(j)
        # np.random.uniform(-0.25, 0.25)
        data_tx.append(doc_vec[j] / 1 if doc_len==0 else doc_len)  # doc_vec[j] / doc_len

# tx = sp.csr_matrix((test_size, word_embeddings_dim), dtype=np.float32)
tx = sp.csr_matrix((data_tx, (row_tx, col_tx)),
                   shape=(test_size, word_embeddings_dim))

ty = []
for i in range(test_size):
    doc_meta = shuffle_doc_name_list[i + train_size]
    temp = doc_meta.split('\t')
    one_hot = [0 for l in range(len(label_list))]
    for label in temp[2].split(' '):
        label_index = label_list.index(label)
        one_hot[label_index] = 1

    ty.append(one_hot)
ty = np.array(ty)
# print(ty)


# In[11]:



# allx: the the feature vectors of both labeled and unlabeled training instances
# (a superset of x)
# unlabeled training instances -> words

word_vectors = np.random.uniform(-0.01, 0.01,
                                 (vocab_size, word_embeddings_dim))

for i in range(len(vocab)):
    word = vocab[i]
    if word in word_vector_map:
        vector = word_vector_map[word]
        word_vectors[i] = vector

row_allx = []
col_allx = []
data_allx = []

for i in range(train_size):
    doc_vec = np.array([0.0 for k in range(word_embeddings_dim)])
    doc_words = shuffle_doc_words_list[i]
    words = doc_words.split()
    doc_len = len(words)
    for word in words:
        if word in word_vector_map:
            word_vector = word_vector_map[word]
            doc_vec = doc_vec + np.array(word_vector)

    for j in range(word_embeddings_dim):
        row_allx.append(int(i))
        col_allx.append(j)
        # np.random.uniform(-0.25, 0.25)
        data_allx.append(doc_vec[j] / 1 if doc_len==0 else doc_len)  # doc_vec[j]/doc_len
for i in range(vocab_size):
    for j in range(word_embeddings_dim):
        row_allx.append(int(i + train_size))
        col_allx.append(j)
        data_allx.append(word_vectors.item((i, j)))


row_allx = np.array(row_allx)
col_allx = np.array(col_allx)
data_allx = np.array(data_allx)

allx = sp.csr_matrix(
    (data_allx, (row_allx, col_allx)), shape=(train_size + vocab_size, word_embeddings_dim))

ally = []
for i in range(train_size):
    doc_meta = shuffle_doc_name_list[i]
    temp = doc_meta.split('\t')
    one_hot = [0 for l in range(len(label_list))]
    for label in temp[2].split(' '):
        label_index = label_list.index(label)
        one_hot[label_index] = 1
    ally.append(one_hot)    
    

for i in range(vocab_size):
    one_hot = [0 for l in range(len(label_list))]
    ally.append(one_hot)

ally = np.array(ally)

print(x.shape, y.shape, tx.shape, ty.shape, allx.shape, ally.shape)


# In[ ]:



'''
Doc word heterogeneous graph
'''

# word co-occurence with context windows
window_size = 20
windows = []

for doc_words in shuffle_doc_words_list:
    words = doc_words.split()
    length = len(words)
    if length <= window_size:
        windows.append(words)
    else:
        # print(length, length - window_size + 1)
        for j in range(length - window_size + 1):
            window = words[j: j + window_size]
            windows.append(window)
            # print(window)


word_window_freq = {}
for window in windows:
    appeared = set()
    for i in range(len(window)):
        if window[i] in appeared:
            continue
        if window[i] in word_window_freq:
            word_window_freq[window[i]] += 1
        else:
            word_window_freq[window[i]] = 1
        appeared.add(window[i])

word_pair_count = {}
for window in windows:
    for i in range(1, len(window)):
        for j in range(0, i):
            word_i = window[i]
            word_i_id = word_id_map[word_i]
            word_j = window[j]
            word_j_id = word_id_map[word_j]
            if word_i_id == word_j_id:
                continue
            word_pair_str = str(word_i_id) + ',' + str(word_j_id)
            if word_pair_str in word_pair_count:
                word_pair_count[word_pair_str] += 1
            else:
                word_pair_count[word_pair_str] = 1
            # two orders
            word_pair_str = str(word_j_id) + ',' + str(word_i_id)
            if word_pair_str in word_pair_count:
                word_pair_count[word_pair_str] += 1
            else:
                word_pair_count[word_pair_str] = 1

row = []
col = []
weight = []

# pmi as weights

num_window = len(windows)

for key in word_pair_count:
    temp = key.split(',')
    i = int(temp[0])
    j = int(temp[1])
    count = word_pair_count[key]
    word_freq_i = word_window_freq[vocab[i]]
    word_freq_j = word_window_freq[vocab[j]]
    pmi = log((1.0 * count / num_window) /
              (1.0 * word_freq_i * word_freq_j/(num_window * num_window)))
    if pmi <= 0:
        continue
    row.append(train_size + i)
    col.append(train_size + j)
    weight.append(pmi)


# In[ ]:


row1 = []
col1 = []
weight1 = []

# pmi as weights

num_window = len(windows)

for key in word_pair_count:
    temp = key.split(',')
    i = int(temp[0])
    j = int(temp[1])
    count = word_pair_count[key]
    word_freq_i = word_window_freq[vocab[i]]
    word_freq_j = word_window_freq[vocab[j]]
    pmi = log((1.0 * count / num_window) /
              (1.0 * word_freq_i * word_freq_j/(num_window * num_window)))
    if pmi <= 0:
        continue
    row.append(train_size + i)
    col.append(train_size + j)
    weight.append(pmi)


# In[ ]:



# word vector cosine similarity as weights

# doc word frequency
doc_word_freq = {}

for doc_id in range(len(shuffle_doc_words_list)):
    doc_words = shuffle_doc_words_list[doc_id]
    words = doc_words.split()
    for word in words:
        word_id = word_id_map[word]
        doc_word_str = str(doc_id) + ',' + str(word_id)
        if doc_word_str in doc_word_freq:
            doc_word_freq[doc_word_str] += 1
        else:
            doc_word_freq[doc_word_str] = 1

for i in range(len(shuffle_doc_words_list)):
    doc_words = shuffle_doc_words_list[i]
    words = doc_words.split()
    doc_word_set = set()
    for word in words:
        if word in doc_word_set:
            continue
        j = word_id_map[word]
        key = str(i) + ',' + str(j)
        freq = doc_word_freq[key]
        if i < train_size:
            row.append(i)
        else:
            row.append(i + vocab_size)
        col.append(train_size + j)
        idf = log(1.0 * len(shuffle_doc_words_list) /
                  word_doc_freq[vocab[j]])
        weight.append(freq * idf)
        doc_word_set.add(word)


# In[ ]:


node_size = train_size + vocab_size + test_size
adj = sp.csr_matrix(
    (weight, (row, col)), shape=(node_size, node_size))


# In[ ]:


print('windows',len(windows))
print('word_pair_count',len(word_pair_count))
print('word_window_freq',len(word_window_freq))
print('weight',len(weight)) 
print('row',len(row)) 
print('col',len(col)) 
print('train_size',train_size) 
print('vocab_size',vocab_size) 
print('test_size',test_size)
print('node_size',node_size)


# In[ ]:



# dump objects
dataset='all'
f = open(Path+"/ind.{}.x".format(dataset), 'wb')
pkl.dump(x, f)
f.close()

f = open(Path+"/ind.{}.y".format(dataset), 'wb')
pkl.dump(y, f)
f.close()

f = open(Path+"/ind.{}.tx".format(dataset), 'wb')
pkl.dump(tx, f)
f.close()

f = open(Path+"/ind.{}.ty".format(dataset), 'wb')
pkl.dump(ty, f)
f.close()

f = open(Path+"/ind.{}.allx".format(dataset), 'wb')
pkl.dump(allx, f)
f.close()

f = open(Path+"/ind.{}.ally".format(dataset), 'wb')
pkl.dump(ally, f)
f.close()

f = open(Path+"/ind.{}.adj".format(dataset), 'wb')
pkl.dump(adj, f)
f.close()


# In[ ]:




