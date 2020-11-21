## Task-Oriented Dialog Generation with Enhanced Entity Representation

This is the PyTorch implementation of the paper:
**Task-Oriented Dialog Generation with Enhanced Entity Representation**. [**Zhenhao He**], [Jiachun Wang], [Jian Chen]. ***INTERSPEECH 2020***.


This code has been written using PyTorch >= 0.4. If you use any source codes included this toolkit in your work, please cite the following paper. The bibtex are listed below:
<pre>
@inproceedings{he2020fg2seq,
  title={Task-Oriented Dialog Generation with Enhanced Entity Representation},
  author={He, Zhenhao and Wang, Jiachun and Chen Jian},
  booktitle={ INTERSPEECH},
  year={2020}
}
</pre>


## Preprocessing the datasets
We created `preprocessing.sh` to preprocess the datasets. You can run:
```console
❱❱❱ ./preprocessing.sh
```

## Train a model for task-oriented dialog datasets
We created `myTrain.py` to train models. You can run:
FG2SEQ InCar:
```console
❱❱❱ python myTrain.py -lr=0.001 -hdd=128 -dr=0.2 -bsz=16 -ds=kvr -B=10 -ss=10.0
```
or FG2SEQ CamRest:
```console
❱❱❱ python myTrain.py -lr=0.001 -hdd=128 -dr=0.2 -bsz=8 -ds=cam -B=5 -ss=10.0
```

While training, the model with the best validation is saved. If you want to reuse a model add `-path=path_name_model` to the function call. The model is evaluated by using F1 and BLEU.

## Test a model for task-oriented dialog datasets
We created  `myTest.py` to test models. You can run:
```console
❱❱❱ python myTest.py -path=<path_to_saved_model> 
```

## Acknowledgement

**Global-to-local Memory Pointer Networks for Task-Oriented Dialogue**. [Chien-Sheng Wu](https://jasonwu0731.github.io/), [Richard Socher](https://www.socher.org/), [Caiming Xiong](http://www.stat.ucla.edu/~caiming/). ***ICLR 2019***. [[PDF]](https://arxiv.org/abs/1901.04713) [[Open Reivew]](https://openreview.net/forum?id=ryxnHhRqFm) [[Code]](https://github.com/jasonwu0731/GLMP)

>   We are highly grateful for the public code of GLMP!