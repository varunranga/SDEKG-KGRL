#!/bin/bash

python3 main.py \
-ds "NELL995" \
-em "TransE" \
-es 50 \
-bs 128 \
-mg 1.0 \
-lr 0.001 \
-pt 50 \
-st "bernoulli" \
-sd "Dataset_NELL995_TransE" \
-te \
-ns 15000000 \
-pc \
-sf "PlotCurves_NELL995_TransE.png" \
-ee \
-nw 10000 \
-wl 1 2 3 5 10 20 \
-se "Environment_NELL995_TransE" \
-sr

python3 main.py \
-ds "NELL995" \
-em "TransH" \
-es 50 \
-bs 128 \
-mg 1.0 \
-lr 0.001 \
-pt 50 \
-st "bernoulli" \
-sd "Dataset_NELL995_TransH" \
-te \
-ns 15000000 \
-pc \
-sf "PlotCurves_NELL995_TransH.png" \
-ee \
-nw 10000 \
-wl 1 2 3 5 10 20 \
-se "Environment_NELL995_TransH" \
-sr

python3 main.py \
-ds "NELL995" \
-em "TransD" \
-es 50 \
-bs 128 \
-mg 1.0 \
-lr 0.001 \
-pt 50 \
-st "bernoulli" \
-sd "Dataset_NELL995_TransD" \
-te \
-ns 15000000 \
-pc \
-sf "PlotCurves_NELL995_TransD.png" \
-ee \
-nw 10000 \
-wl 1 2 3 5 10 20 \
-se "Environment_NELL995_TransD" \
-sr