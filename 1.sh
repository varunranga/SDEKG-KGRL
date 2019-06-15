#!/bin/bash

python3 main.py \
-ds "FB15K" \
-em "TransE" \
-es 50 \
-bs 256 \
-mg 1.0 \
-lr 0.001 \
-pt 50 \
-st "bernoulli" \
-sd "Dataset_FB15K_TransE" \
-te \
-ns 15000000 \
-pc \
-sf "PlotCurves_FB15K_TransE.png" \
-ee \
-nw 10000 \
-wl 1 2 3 5 10 20 \
-se "Environment_FB15K_TransE" \
-sr

python3 main.py \
-ds "FB15K" \
-em "TransH" \
-es 50 \
-bs 256 \
-mg 1.0 \
-lr 0.001 \
-pt 50 \
-st "bernoulli" \
-sd "Dataset_FB15K_TransH" \
-te \
-ns 15000000 \
-pc \
-sf "PlotCurves_FB15K_TransH.png" \
-ee \
-nw 10000 \
-wl 1 2 3 5 10 20 \
-se "Environment_FB15K_TransH" \
-sr

python3 main.py \
-ds "FB15K" \
-em "TransD" \
-es 50 \
-bs 256 \
-mg 1.0 \
-lr 0.001 \
-pt 50 \
-st "bernoulli" \
-sd "Dataset_FB15K_TransD" \
-te \
-ns 15000000 \
-pc \
-sf "PlotCurves_FB15K_TransD.png" \
-ee \
-nw 10000 \
-wl 1 2 3 5 10 20 \
-se "Environment_FB15K_TransD" \
-sr

