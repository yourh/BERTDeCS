#!/usr/bin/env bash

for ((i=0; i<5; i++)); do
  torchrun --nproc-per-node 4 main.py \
  configures/data.yaml \
  configures/BERTDeCS-CL.yaml \
  --data DeCS_CL \
  --train-name train_cl \
  --train \
  --model-id "${i}" \
  --dist -a

  torchrun --nproc-per-node 4 main.py \
  configures/data.yaml \
  configures/BERTDeCS-Af.yaml \
  --data DeCS_PM300W \
  --labels decs_mesh \
  --train-name train_300w \
  --train-labels mesh \
  --valid-name dev_st1 \
  --valid-labels decs \
  --train \
  --model-id CL_"${i}" \
  -p models/BERTDeCS_CL-DeCS_CL-Model_"${i}".pt \
  -b 25 \
  --dist -a

  torchrun --nproc-per-node 4 main.py \
  configures/data.yaml \
  configures/BERTDeCS-A.yaml \
  --train-name train_es \
  --valid-name dev_st1 \
  --train --eval "dev_st1,test_st1" \
  --model-id "CL_PM_${i}" \
  -p models/BERTDeCS_Af-DeCS_PM300W-Model_CL_"${i}".pt \
  -b 25 \
  --dist -a
done
