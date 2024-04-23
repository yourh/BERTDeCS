#!/usr/bin/env bash

for ((i=0; i< 5; i++)); do
  torchrun --nproc-per-node 4 main.py \
  configures/data.yaml \
  configures/BERTDeCS-CL.yaml \
  --train-name train_cl \
  --train \
  --model-id "${i}" \
  --dist -a

  torchrun --nproc-per-node 4 main.py \
  configures/data.yaml \
  configures/BERTDeCS-Af.yaml \
  --train-name train_pubmed \
  --valid-name dev_st1 \
  --labels mesh_decs \
  --train \
  --model-id CL_"${i}" \
  -p models/BERTDeCS_CL-DeCS_CL-Model_"${i}".pt \
  -b 25 \
  --dist -a

  torchrun --nproc-per-node 4 main.py \
  configures/data.yaml \
  configures/BERTDeCS.yaml \
  --train-name train_es \
  --valid-name dev_st1 \
  --labels decs \
  --train --eval "dev_st1,test_st1" \
  --model-id "CL_PM_${i}" \
  -p models/BERTDeCS_Af-DeCS_PM300W-Model_CL_"${i}".pt \
  -b 25 \
  --dist -a
done
