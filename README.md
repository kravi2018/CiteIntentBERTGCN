# CiteIntentBERTGCN
This repo contains code for [CiteIntentBERTGCN: Multi-sentence and multi-label based citation intent classification using BERT and graphical convolution neural network].

## Usage

1. Run `run_GenerateGraph_Paper.sh` to generate dataset with different combination of context lenght and support and build the text graph.

2. Run `python finetune_bert.py --dataset [dataset]` 
to finetune the BERT model over target dataset. The model and training logs will be saved to `checkpoint/[bert_init]_[dataset]/` by default. 
Run `python finetune_bert.py -h` to see the full list of hyperparameters.

3. Run `run_BertGCN_multilabel_Paper.sh` to train the BertGCN. 

