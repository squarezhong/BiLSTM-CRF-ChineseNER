# BiLSTM-CRF-ChineseNER
PyTorch implementation of BiLSTM-CRF for Chinese NER

## Requirement: 
- Python 3.12.2
- PyTorch 2.2.1
- Numpy 1.26.4

- Perl 5.38.2

Other versions may also work, but I didn't test.

## Usage:
Just run `python main.py --mode=train` 

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


Here's the result at the end of training (32ed epoch):
```shell
processed 13563 tokens with 458 phrases; found: 403 phrases; correct: 332.
accuracy:  96.79%; precision:  82.38%; recall:  72.49%; FB1:  77.12
              LOC: precision:  84.56%; recall:  67.74%; FB1:  75.22  149
              ORG: precision:  78.32%; recall:  67.47%; FB1:  72.49  143
              PER: precision:  84.68%; recall:  88.68%; FB1:  86.64  111
```

## TODO
- [ ] Add more datasets.
- [ ] Use python to calculate metrics.
- [ ] Visualize the results.
- [ ] Fine-tune the hyper-parameters.
