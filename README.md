# BERTDeCS
Large-scale semantic indexing of Spanish biomedical literature using contrastive transfer learning

## Quick Start
Install the requirements of BERTDeCS:
```bash
git clone https://github.com/yourh/BERTDeCS.git
cd BERTDeCS
conda create -n BERTDeCS python=3.12
conda activate BERTDeCS
pip install -r requirements.txt
mkdir models
cd models
wget https://zenodo.org/records/14190447/files/BERTDeCS_A-DeCS_ES.pt
cd ..
```

Preprocess the citations with journal names, titles and abstracts:
```bash
python preprocess.py tokenize \
-j data/test_st1_journal.txt \
-t data/test_st1_title.txt \
-a data/test_st1_abstract.txt \
-o data/test_st1
```

Predict the DeCS terms by BERTDeCS:
```bash
python main.py \
configures/data.yaml \
configures/BERTDeCS-A.yaml \
--eval "test_st1" \
-b 25 \
-a
```

Evaluate the performance of prediction:
```bash
python evaluation.py \
-t data/test_st1_decs.txt \
-r results/BERTDeCS_A-DeCS_ES-test_st1.npz \
-n 10
```

## Training
We have trained BERTDeCS on 4×4090 by following steps:
1. Preprocess the pre-training and training data by
```bash
python preprocess.py tokenize \
-j data/{journal} \
-t data/{title} \
-a data/{abstract} \
-o data/{data_name}
```
2. Run contrastive learning by
```bash
torchrun --nproc-per-node 4 main.py \
configures/data.yaml \
configures/BERTDeCS-CL.yaml \
--data DeCS_CL \
--train-name train_cl \
--train \
--dist -a
```
3. Run pre-training by
```bash
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
-p models/BERTDeCS_CL-DeCS_CL.pt \
-b 25 \
--dist -a
```

4. Run fine-tuning by
```bash
torchrun --nproc-per-node 4 main.py \
configures/data.yaml \
configures/BERTDeCS-A.yaml \
--train-name train_es \
--valid-name dev_st1 \
--train --eval "dev_st1,test_st1" \
-p models/BERTDeCS_Af-DeCS_PM300W.pt \
-b 25 \
--dist -a
```

## Reference

## Declaration
It is free for non-commercial use. For commercial use, please contact Dr. Ronghui You and Prof. Shanfeng Zhu (<zhusf@fudan.edu.cn>).