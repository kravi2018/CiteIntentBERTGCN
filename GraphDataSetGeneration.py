#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import json
import os


# In[2]:


from utils import loadWord2Vec, clean_str
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--supportTxt', type=str, default='SizeAll', choices=['Size1', 'Size2', 'Size3', 'Size4', 'SizeAll'])
parser.add_argument('--contextDB', type=str, default='classification_gold_context', choices=['classification_gold_context','classification_1_context','classification_3_context','classification_5_context','classification_7_context','classification_9_context'])

# In[3]:
args = parser.parse_args()
contextDB = args.contextDB
support= args.supportTxt

size=0
if support=="Size1":
    size= 1
elif support=="Size2":
    size= 2
elif support=="Size3":
    size= 3
elif support=="Size4":
    size= 4
else:
    size=0


# contextDB="classification_gold_context"
# support='SizeAll' #'Size1', 'Size2', 'Size3','Size4', 'SizeAll'
# size=0          #1,2,3,4,0

if not os.path.exists("data/"+contextDB+"/"+support):
    os.mkdir("data/"+contextDB+"/"+support)
    
if not os.path.exists("data/"+contextDB+"/"+support+"/corpus"):
    os.mkdir("data/"+contextDB+"/"+support+"/corpus")

#dev
f = open(r"data/"+contextDB+"/dev.json")
data = json.load(f)
dev = pd.DataFrame(data)
# dev=dev[:4]
dev['x'].to_csv("data/"+contextDB+"/"+support+"/corpus/dev.txt", header = None, index = False, sep ='|')

dev.insert(0, 'Env', 'val')
dev.drop(dev.columns[[1, 2]], axis=1, inplace=True)
dev.to_csv("data/"+contextDB+"/"+support+"/dev.txt", header = None, sep ='\t')


# In[4]:


#train
f = open(r"data/"+contextDB+"/train.json")
data = json.load(f)
train = pd.DataFrame(data)
# train=train[:24]
train['x'].to_csv("data/"+contextDB+"/"+support+"/corpus/train.txt", header = None, index = False, sep ='|')

train.insert(0, 'Env', 'train')
train.drop(train.columns[[1, 2]], axis=1, inplace=True)
train.to_csv("data/"+contextDB+"/"+support+"/train.txt", header = None, sep ='\t')


# In[5]:


#test
f = open(r"data/classification_gold_context/testwithsupport.json")
data = json.load(f)
test = pd.DataFrame(data)
if(size !=0):
    test = test[(test.gold_length == size)]
# test=test[:12]
test['x'].to_csv("data/"+contextDB+"/"+support+"/corpus/test.txt", header = None, index = False, sep ='|')

test.insert(0, 'Env', 'test')
test.drop(test.columns[[1, 2, 4]], axis=1, inplace=True)
test.to_csv("data/"+contextDB+"/"+support+"/test.txt", header = None, sep ='\t')


# In[6]:


frames = [train, dev, test]
dfall = pd.concat(frames).reset_index()
dfall.drop(dfall.columns[[0]], axis=1, inplace=True)
dfall.to_csv("data/"+contextDB+"/"+support+"/all.txt", header = None, sep ='\t')


# In[7]:


doc_content_list_all = []

f = open('data/'+contextDB+'/'+support+'/corpus/train.txt', encoding="utf8")
lines = f.readlines()
for line in lines:
    doc_content_list_all.append(clean_str(line.strip()))
f.close()

f = open('data/'+contextDB+'/'+support+'/corpus/dev.txt', encoding="utf8")
lines = f.readlines()
for line in lines:
    doc_content_list_all.append(clean_str(line.strip()))
f.close()

f = open('data/'+contextDB+'/'+support+'/corpus/test.txt', encoding="utf8")
lines = f.readlines()
for line in lines:
    doc_content_list_all.append(clean_str(line.strip()))
f.close()

doc_content_list_all_str = '\n'.join(doc_content_list_all)
f = open('data/'+contextDB+'/'+support+'/corpus/all.clean.txt', 'w')
f.write(doc_content_list_all_str)
f.close()


devGroup=dfall.groupby(['Env','y']).size()
print(devGroup)






