#!/bin/bash

python3 main.py \
-ds "Countries" \
-em "TransR" \
-es 25 \
-bs 256 \
-mg 1.0 \
-lr 0.001 \
-pt 50 \
-st "bernoulli" \
-sd "Dataset_Countries_TransR" \
-te \
-ns 500000 \
-pc \
-sf "PlotCurves_Countries_TransR.png" \
-ee \
-nw 500 \
-wl 1 2 3 5 \
-se "Environment_Countries_TransR" \
-sr

python3 main.py \
-ds "UMLS" \
-em "TransR" \
-es 50 \
-bs 256 \
-mg 1.0 \
-lr 0.001 \
-pt 50 \
-st "bernoulli" \
-sd "Dataset_UMLS_TransR" \
-te \
-ns 500000 \
-pc \
-sf "PlotCurves_UMLS_TransR.png" \
-ee \
-nw 500 \
-wl 1 2 3 5 \
-se "Environment_UMLS_TransR" \
-sr

python3 main.py \
-ds "Kinship" \
-em "TransR" \
-es 50 \
-bs 512 \
-mg 1.0 \
-lr 0.001 \
-pt 50 \
-st "bernoulli" \
-sd "Dataset_Kinship_TransR" \
-te \
-ns 500000 \
-pc \
-sf "PlotCurves_Kinship_TransR.png" \
-ee \
-nw 500 \
-wl 1 2 3 5 \
-se "Environment_Kinship_TransR" \
-sr

python3 main.py \
-ds "Countries" \
-em "TransD" \
-es 25 \
-bs 256 \
-mg 1.0 \
-lr 0.001 \
-pt 50 \
-st "bernoulli" \
-sd "Dataset_Countries_TransD" \
-te \
-ns 500000 \
-pc \
-sf "PlotCurves_Countries_TransD.png" \
-ee \
-nw 500 \
-wl 1 2 3 5 \
-se "Environment_Countries_TransD" \
-sr

python3 main.py \
-ds "UMLS" \
-em "TransD" \
-es 50 \
-bs 256 \
-mg 1.0 \
-lr 0.001 \
-pt 50 \
-st "bernoulli" \
-sd "Dataset_UMLS_TransD" \
-te \
-ns 500000 \
-pc \
-sf "PlotCurves_UMLS_TransD.png" \
-ee \
-nw 500 \
-wl 1 2 3 5 \
-se "Environment_UMLS_TransD" \
-sr

python3 main.py \
-ds "Kinship" \
-em "TransD" \
-es 50 \
-bs 512 \
-mg 1.0 \
-lr 0.001 \
-pt 50 \
-st "bernoulli" \
-sd "Dataset_Kinship_TransD" \
-te \
-ns 500000 \
-pc \
-sf "PlotCurves_Kinship_TransD.png" \
-ee \
-nw 500 \
-wl 1 2 3 5 \
-se "Environment_Kinship_TransD" \
-sr

python3 main.py \
-ds "FB15K" \
-em "TransR" \
-es 50 \
-bs 256 \
-mg 1.0 \
-lr 0.001 \
-pt 50 \
-st "bernoulli" \
-sd "Dataset_FB15K_TransR" \
-te \
-ns 15000000 \
-pc \
-sf "PlotCurves_FB15K_TransR.png" \
-ee \
-nw 10000 \
-wl 1 2 3 5 10 20 \
-se "Environment_FB15K_TransR" \
-sr