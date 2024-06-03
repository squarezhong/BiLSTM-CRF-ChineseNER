# BiLSTM-CRF-ChineseNER

[English](README.md) | [简体中文](README_cn.md)

## Introduction

PyTorch implementation of BiLSTM-CRF for Chinese NER.

**CPU, CUDA, and MPS** are supported.

For the principle of BiLSTM-CRF, you can refer to the paper [Bidirectional LSTM-CRF Models for Sequence Tagging](https://arxiv.org/abs/1508.01991).

You can also refer to the blog [一文读懂BiLSTM+CRF实现命名实体识别](https://paddlepedia.readthedocs.io/en/latest/tutorials/natural_language_processing/ner/bilstm_crf.html) for more details.

## Requirement: 
- Python 3.12.2
- PyTorch 2.2.1
- Numpy 1.26.4

- Perl 5.38.2

Other versions may also work, but I didn't test.

## Usage:
Just run `python main.py --mode=train` to start training.

If you want to use GPU, you can add `--use-cuda` or `--use-mps` to the command.

Use `python main.py --help` to see the full list of parameters.
```
usage: main.py [-h] [--epochs EPOCHS] [--batch-size BATCH_SIZE] [--seed SEED] [--use-cuda] [--use-mps] [--lr LR] [--use-crf] [--mode MODE] [--save SAVE]
               [--save-epoch] [--data DATA] [--word-ebd-dim WORD_EBD_DIM] [--dropout DROPOUT] [--lstm-hsz LSTM_HSZ] [--lstm-layers LSTM_LAYERS] [--l2 L2]
               [--clip CLIP] [--result-path RESULT_PATH]

LSTM_CRF

options:
  -h, --help            show this help message and exit
  --epochs EPOCHS       number of epochs for train
  --batch-size BATCH_SIZE
                        batch size for training
  --seed SEED           random seed
  --use-cuda            enables cuda
  --use-mps             enables mps
  --lr LR               learning rate
  --use-crf             use crf
  --mode MODE           train mode or test mode
  --save SAVE           path to save the final model
  --save-epoch          save every epoch
  --data DATA           location of the data corpus
  --word-ebd-dim WORD_EBD_DIM
                        number of word embedding dimension
  --dropout DROPOUT     the probability for dropout
  --lstm-hsz LSTM_HSZ   BiLSTM hidden size
  --lstm-layers LSTM_LAYERS
                        biLSTM layer numbers
  --l2 L2               l2 regularization
  --clip CLIP           gradient clipping
  --result-path         result-path
```

P.S. use standard `conlleval.pl` to calculate entity-level precision, recall and f1-score. For details, you can refer the source codes. 


Here's the result at the end of training (64ed epoch):
```shell
processed 13563 tokens with 458 phrases; found: 437 phrases; correct: 356.
accuracy:  97.10%; 
ALL: precision:  81.46%; recall:  77.73%; FB1:  79.55  Num: 437
LOC: precision:  79.10%; recall:  75.27%; FB1:  77.13  Num: 177
ORG: precision:  75.62%; recall:  72.89%; FB1:  74.23  Num: 160
PER: precision:  95.00%; recall:  89.62%; FB1:  92.23  Num: 100
```

## TODO
- [ ] Add more datasets.
- [ ] Use python to calculate metrics.
- [ ] Visualize the results.
- [ ] Fine-tune the hyper-parameters.
