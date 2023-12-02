#!/usr/bin/env bash

for m in 0.2; do
    for gcn_model in "gcn"; do
        for EPOCHS in 10; do
            for dropout in 0.4; do
                for supportTxt in "Size2"; do
                    python train_bert_gcn.py \
                      --m ${m} \
                      --nb_epochs ${EPOCHS} \
                      --dropout ${dropout} \
                      --gcn_model ${gcn_model} \
                      --supportTxt ${supportTxt}
                  done;
            done;
        done;
    done;
done;

for m in 0.2; do
    for gcn_model in "gat"; do
        for EPOCHS in 10; do
            for dropout in 0.6; do
                for supportTxt in "Size2"; do
                    python train_bert_gcn.py \
                      --m ${m} \
                      --nb_epochs ${EPOCHS} \
                      --dropout ${dropout} \
                      --gcn_model ${gcn_model} \
                      --supportTxt ${supportTxt}
                  done;
            done;
        done;
    done;
done;