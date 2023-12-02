#!/usr/bin/env bash

for contextDB in "classification_1_context" ; do
    for supportTxt in "Size1" "Size2" "Size3" "Size4" "SizeAll"; do  
      
            python GraphDataSetGeneration.py \
              --contextDB ${contextDB} \
              --supportTxt ${supportTxt} 
			  
	    python build_graph.py \
              --contextDB ${contextDB} \
              --supportTxt ${supportTxt} 
			                 
    done;
done;
