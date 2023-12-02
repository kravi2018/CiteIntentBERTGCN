#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch as th
from transformers import AutoModel, AutoTokenizer
import torch.nn.functional as F

from utils import *
import dgl
import torch.utils.data as Data
from ignite.engine import Events, create_supervised_evaluator, create_supervised_trainer, Engine
from ignite.metrics import Accuracy, Loss
import numpy as np
import os
from datetime import datetime
from sklearn.metrics import accuracy_score
import argparse, shutil, logging
from torch.optim import lr_scheduler
from model import BertClassifier


# In[2]:


from torch.nn import BCEWithLogitsLoss  #change
from multilabel_evaluation import evaluate as evaluate_multilabel


# In[3]:

# Seed
seed = 123
th.manual_seed(seed)
th.cuda.manual_seed(seed)
th.cuda.manual_seed_all(seed)
np.random.seed(seed)

parser = argparse.ArgumentParser()
parser.add_argument('--max_length', type=int, default=128, help='the input length for bert')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--nb_epochs', type=int, default=15) #change
parser.add_argument('--bert_lr', type=float, default=1e-5)
parser.add_argument('--dataset', default='all', choices=['all','20ng', 'R8', 'R52', 'ohsumed', 'mr'])  #change
parser.add_argument('--bert_init', type=str, default='roberta-large',
                    choices=['roberta-base', 'roberta-large', 'bert-base-uncased', 'bert-large-uncased'])
parser.add_argument('--checkpoint_dir', default=None, help='checkpoint directory, [bert_init]_[dataset] if not specified')


# In[4]:


args = parser.parse_args("")

max_length = args.max_length
batch_size = args.batch_size
nb_epochs = args.nb_epochs
bert_lr = args.bert_lr
dataset = args.dataset
bert_init = args.bert_init
checkpoint_dir = args.checkpoint_dir
if checkpoint_dir is None:
    ckpt_dir = './checkpoint/{}_{}_{}'.format(bert_init, dataset,bert_lr)
else:
    ckpt_dir = checkpoint_dir

os.makedirs(ckpt_dir, exist_ok=True)
# shutil.copy(os.path.basename(__file__), ckpt_dir)  #change commented

sh = logging.StreamHandler(sys.stdout)
sh.setFormatter(logging.Formatter('%(message)s'))
sh.setLevel(logging.INFO)
fh = logging.FileHandler(filename=os.path.join(ckpt_dir, 'training.log'), mode='w')
fh.setFormatter(logging.Formatter('%(message)s'))
fh.setLevel(logging.INFO)
logger = logging.getLogger('training logger')
logger.addHandler(sh)
logger.addHandler(fh)
logger.setLevel(logging.INFO)

cpu = th.device('cpu')
#gpu = th.device('cuda:0')
gpu = th.device("cuda:3" if th.cuda.is_available() else "cpu")  #change

logger.info('arguments:')
logger.info(str(args))
logger.info('checkpoints will be saved in {}'.format(ckpt_dir))

# Data Preprocess
adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, train_size, test_size = load_corpus(dataset,'classification_gold_context/SizeAll')
'''
y_train, y_val, y_test: n*c matrices 
train_mask, val_mask, test_mask: n-d bool array
train_size, test_size: unused
'''

# compute number of real train/val/test/word nodes and number of classes
nb_node = adj.shape[0]
nb_train, nb_val, nb_test = train_mask.sum(), val_mask.sum(), test_mask.sum()
nb_word = nb_node - nb_train - nb_val - nb_test
nb_class = y_train.shape[1]

# instantiate model according to class number
model = BertClassifier(pretrained_model=bert_init, nb_class=nb_class)  #no change, this is without GCN

# # transform one-hot label to class ID for pytorch computation
# y = th.LongTensor((y_train + y_val +y_test).argmax(axis=1))
# issue - look like only one class as output


# In[5]:


# #y = np.array([np.argwhere(x > 0.5) for x in y])  # cross-check np.array()
# #y[:6]
# y = y_train + y_val +y_test
# y1 = y.argmax(axis=1)
# print(y1.shape)

# y2=np.where(y<0, 0, 1)
# #y2 = y[y<0] = 0#np.asarray([np.argwhere(x > 0.5) for x in y])
# print(y2.shape)

# y = y_train + y_val +y_test
# # y = np.asarray([np.argwhere(x > 0.5) for x in y])
# y1 = th.LongTensor(y1)
# y2 = th.LongTensor(y2)
y = y_train + y_val +y_test
y2 = np.where(y<0.5, 0, 1)
y=th.from_numpy(y2)  #change

# y1 = y.argmax(axis=1)
# y1=th.LongTensor(y1)
label = {}
label['train'], label['val'], label['test'] = y[:nb_train], y[nb_train:nb_train+nb_val], y[-nb_test:]


# In[9]:


# load documents and compute input encodings
corpus_file = './data/classification_gold_context/SizeAll/corpus/'+dataset+'_shuffle.txt'
with open(corpus_file, 'r',encoding="latin1") as f:
    text = f.read()
    text=text.replace('\\', '')
    text = text.split('\n')

def encode_input(text, tokenizer):
    input = tokenizer(text, max_length=max_length, truncation=True, padding=True, return_tensors='pt')
    return input.input_ids, input.attention_mask

input_ids, attention_mask = {}, {}

input_ids_, attention_mask_ = encode_input(text, model.tokenizer)

# create train/test/val datasets and dataloaders
input_ids['train'], input_ids['val'], input_ids['test'] =  input_ids_[:nb_train], input_ids_[nb_train:nb_train+nb_val], input_ids_[-nb_test:]
attention_mask['train'], attention_mask['val'], attention_mask['test'] =  attention_mask_[:nb_train], attention_mask_[nb_train:nb_train+nb_val], attention_mask_[-nb_test:]


# In[17]:


datasets = {}
loader = {}
for split in ['train', 'val', 'test']:
    datasets[split] =  Data.TensorDataset(input_ids[split], attention_mask[split], label[split])
    loader[split] = Data.DataLoader(datasets[split], batch_size=batch_size, shuffle=True)
    
# Training

optimizer = th.optim.Adam(model.parameters(), lr=bert_lr)
scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[30], gamma=0.1)    


# In[18]:


all_labels = [] #change
f = open('./data/classification_gold_context/SizeAll/corpus/all_labels.txt', 'r')
lines = f.readlines()
for line in lines:
    all_labels.append(line.strip())
f.close()
all_labels


# In[19]:



def train_step(engine, batch):
    global model, optimizer
    model.train()
    model = model.to(gpu)
    optimizer.zero_grad()
    (input_ids, attention_mask, label) = [x.to(gpu) for x in batch]
    optimizer.zero_grad()
    y_pred = model(input_ids, attention_mask)
    y_true = label.type(th.long)
#     print('y_pred',y_pred)
#     print('y_true',y_true)
    loss_func = BCEWithLogitsLoss()
    loss = loss_func(y_pred, y_true.float())
    loss.backward()
    optimizer.step()
    train_loss = loss.item()
    with th.no_grad():
        y_true = y_true.detach().cpu()
        y_pred = y_pred.detach().cpu() #change
        #y_pred = np.array([np.argwhere(x > 0.5) for x in y_pred])
        y2 = np.where(y_pred<0.5, 0, 1)
        y_pred=th.from_numpy(y2)  #change

        #y_pred = y_pred.argmax(axis=1).detach().cpu()
        #train_acc = accuracy_score(y_true, y_pred)  
        train_acc = evaluate_multilabel(y_pred, y_true, all_labels) #change
        
    return train_acc, train_loss


trainer = Engine(train_step)


def test_step(engine, batch):
    global model
    with th.no_grad():
        model.eval()
        model = model.to(gpu)
        (input_ids, attention_mask, label) = [x.to(gpu) for x in batch]
        optimizer.zero_grad()
        y_pred = model(input_ids, attention_mask)
        y_true = label
        train_acc = evaluate_multilabel(y_pred, y_true, all_labels) #change

        return train_acc

evaluator = Engine(test_step)
# metrics={
#     'acc': evaluate_multilabel(y_pred, y_true, all_labels),
#     'nll': Loss(th.nn.BCEWithLogitsLoss())
# }
# for n, f in metrics.items():
#     f.attach(evaluator, n)

# In[24]:


@trainer.on(Events.EPOCH_COMPLETED)
def log_training_results(trainer):
    evaluator.run(loader['train'])
    #metrics = evaluator.state.metrics
    train_acc =evaluator.state.output
#     train_nll, train_acc =trainer.state.output
    #print('train_acc',train_acc)
    evaluator.run(loader['val'])
    #metrics = evaluator.state.metrics
    val_acc =evaluator.state.output
#     val_nll, val_acc  = trainer.state.output
#     print('val_acc',val_acc)
    evaluator.run(loader['test'])
    #metrics = evaluator.state.metrics
    test_acc =evaluator.state.output
    logger.info(
        "\rEpoch: {}  Train acc: {} Val acc: {} Test acc: {}"
        .format(trainer.state.epoch, train_acc, val_acc, test_acc)
    )
    stricttest_acc = test_acc['strict']
    if stricttest_acc > log_training_results.best_test_acc:
        logger.info("New checkpoint")
        th.save(
            {
		'bert_model': model.bert_model.state_dict(),
                'classifier': model.classifier.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': trainer.state.epoch,
            },
            os.path.join(
                ckpt_dir, 'checkpoint.pth'
            )
        )
        log_training_results.best_test_acc = stricttest_acc


log_training_results.best_test_acc = 0
trainer.run(loader['train'], max_epochs=nb_epochs)


# In[ ]:




