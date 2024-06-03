# BiLSTM-CRF-ChineseNER

[English](README.md) | [简体中文](README_cn.md)

## 介绍

这是一个使用 PyTorch 实现的 BiLSTM-CRF 中文命名实体识别（NER）模型。

**支持 CPU、CUDA 和 MPS**。

关于 BiLSTM-CRF 的原理，可以参考论文 [Bidirectional LSTM-CRF Models for Sequence Tagging](https://arxiv.org/abs/1508.01991)。

你也可以参考博客 [一文读懂BiLSTM+CRF实现命名实体识别](https://paddlepedia.readthedocs.io/en/latest/tutorials/natural_language_processing/ner/bilstm_crf.html) 获取更多详情。

## 环境要求：
- Python 3.12.2
- PyTorch 2.2.1
- Numpy 1.26.4

- Perl 5.38.2

其他版本也可能可以工作，但我没有测试过。

## 使用方法：
只需运行 `python main.py --mode=train` 开始训练。

如果你想使用 GPU，可以在命令中添加 `--use-cuda` 或 `--use-mps`。

使用 `python main.py --help` 查看完整的参数列表。
```
usage: main.py [-h] [--epochs EPOCHS] [--batch-size BATCH_SIZE] [--seed SEED] [--use-cuda] [--use-mps] [--lr LR] [--use-crf] [--mode MODE] [--save SAVE]
               [--save-epoch] [--data DATA] [--word-ebd-dim WORD_EBD_DIM] [--dropout DROPOUT] [--lstm-hsz LSTM_HSZ] [--lstm-layers LSTM_LAYERS] [--l2 L2]
               [--clip CLIP] [--result-path RESULT_PATH]

LSTM_CRF

options:
  -h, --help            显示此帮助信息并退出
  --epochs EPOCHS       训练的轮数
  --batch-size BATCH_SIZE
                        训练的批次大小
  --seed SEED           随机种子
  --use-cuda            启用 CUDA
  --use-mps             启用 MPS
  --lr LR               学习率
  --use-crf             使用 CRF
  --mode MODE           训练模式或测试模式
  --save SAVE           最终模型的保存路径
  --save-epoch          每轮保存
  --data DATA           数据集位置
  --word-ebd-dim WORD_EBD_DIM
                        词嵌入维度数
  --dropout DROPOUT     dropout 概率
  --lstm-hsz LSTM_HSZ   BiLSTM 隐藏层大小
  --lstm-layers LSTM_LAYERS
                        BiLSTM 层数
  --l2 L2               L2 正则化
  --clip CLIP           梯度剪裁
  --result-path         结果路径
```

P.S. 使用标准的 `conlleval.pl` 来计算实体级别的精确度、召回率和 F1 分数。详情请参考源码。

以下是训练结束时（第64轮）的结果：
```shell
processed 13563 tokens with 458 phrases; found: 437 phrases; correct: 356.
accuracy:  97.10%; 
ALL: precision:  81.46%; recall:  77.73%; FB1:  79.55  Num: 437
LOC: precision:  79.10%; recall:  75.27%; FB1:  77.13  Num: 177
ORG: precision:  75.62%; recall:  72.89%; FB1:  74.23  Num: 160
PER: precision:  95.00%; recall:  89.62%; FB1:  92.23  Num: 100
```

## TODO
- [ ] 添加更多数据集。
- [ ] 使用 Python 计算指标。
- [ ] 可视化结果。
- [ ] 微调超参数。