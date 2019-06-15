#!/bin/bash

python3 main.py \
-ds "Countries" \
-em "TransE" \
-es 25 \
-bs 256 \
-mg 1.0 \
-lr 0.001 \
-pt 50 \
-st "bernoulli" \
-sd "Dataset_Countries_TransE" \
-te \
-ns 500000 \
-pc \
-sf "PlotCurves_Countries_TransE.png" \
-ee \
-nw 500 \
-wl 1 2 3 5 \
-se "Environment_Countries_TransE" \
-sr

python3 main.py \
-ds "UMLS" \
-em "TransE" \
-es 50 \
-bs 256 \
-mg 1.0 \
-lr 0.001 \
-pt 50 \
-st "bernoulli" \
-sd "Dataset_UMLS_TransE" \
-te \
-ns 500000 \
-pc \
-sf "PlotCurves_UMLS_TransE.png" \
-ee \
-nw 500 \
-wl 1 2 3 5 \
-se "Environment_UMLS_TransE" \
-sr

python3 main.py \
-ds "Kinship" \
-em "TransE" \
-es 50 \
-bs 512 \
-mg 1.0 \
-lr 0.001 \
-pt 50 \
-st "bernoulli" \
-sd "Dataset_Kinship_TransE" \
-te \
-ns 500000 \
-pc \
-sf "PlotCurves_Kinship_TransE.png" \
-ee \
-nw 500 \
-wl 1 2 3 5 \
-se "Environment_Kinship_TransE" \
-sr

python3 main.py \
-ds "Countries" \
-em "TransH" \
-es 25 \
-bs 256 \
-mg 1.0 \
-lr 0.001 \
-pt 50 \
-st "bernoulli" \
-sd "Dataset_Countries_TransH" \
-te \
-ns 500000 \
-pc \
-sf "PlotCurves_Countries_TransH.png" \
-ee \
-nw 500 \
-wl 1 2 3 5 \
-se "Environment_Countries_TransH" \
-sr

python3 main.py \
-ds "UMLS" \
-em "TransH" \
-es 50 \
-bs 256 \
-mg 1.0 \
-lr 0.001 \
-pt 50 \
-st "bernoulli" \
-sd "Dataset_UMLS_TransH" \
-te \
-ns 500000 \
-pc \
-sf "PlotCurves_UMLS_TransH.png" \
-ee \
-nw 500 \
-wl 1 2 3 5 \
-se "Environment_UMLS_TransH" \
-sr

python3 main.py \
-ds "Kinship" \
-em "TransH" \
-es 50 \
-bs 512 \
-mg 1.0 \
-lr 0.001 \
-pt 50 \
-st "bernoulli" \
-sd "Dataset_Kinship_TransH" \
-te \
-ns 500000 \
-pc \
-sf "PlotCurves_Kinship_TransH.png" \
-ee \
-nw 500 \
-wl 1 2 3 5 \
-se "Environment_Kinship_TransH" \
-sr

python3 main.py \
-ds "NELL995" \
-em "TransR" \
-es 50 \
-bs 128 \
-mg 1.0 \
-lr 0.001 \
-pt 50 \
-st "bernoulli" \
-sd "Dataset_NELL995_TransR" \
-te \
-ns 15000000 \
-pc \
-sf "PlotCurves_NELL995_TransR.png" \
-ee \
-nw 10000 \
-wl 1 2 3 5 10 20 \
-se "Environment_NELL995_TransR" \
-sr