## Task-Oriented Dialog Generation with Enhanced Entity Representation

This is the PyTorch implementation of the paper:
**Task-Oriented Dialog Generation with Enhanced Entity Representation**. ***INTERSPEECH 2020***. [paper link](https://www.isca-speech.org/archive/Interspeech_2020/abstracts/1037.html)


This code has been written using PyTorch >= 0.4. If you use any source codes included this toolkit in your work, please cite the following paper. The bibtex are listed below:
<pre>
@inproceedings{He2020,
  author={Zhenhao He and Jiachun Wang and Jian Chen},
  title={{Task-Oriented Dialog Generation with Enhanced Entity Representation}},
  year=2020,
  booktitle={Proc. Interspeech 2020},
  pages={3905--3909},
  doi={10.21437/Interspeech.2020-1037},
  url={http://dx.doi.org/10.21437/Interspeech.2020-1037}
}
</pre>


## Preprocessing the datasets
To preprocess the dataset, you can run the following command:
```console
❱❱❱ cd data/[KVR|CamRest]
❱❱❱ bash gen.sh
```

## Train a model for task-oriented dialog datasets
We created `myTrain.py` to train models. You can run:
InCar:
```console
❱❱❱ python myTrain.py  -ds kvr -b 8 -hdd 128 -lr 0.001 -dr 0.4
```
or CamRest:
```console
❱❱❱ python myTrain.py  -ds cam -b 8 -hdd 128 -lr 0.001 -dr 0.5
```

While training, the model with the best validation is saved. If you want to reuse a model add `-path=path_name_model` to the function call. The model is evaluated by using F1 and BLEU.

## Test a model for task-oriented dialog datasets
We created  `myTest.py` to test models. You can run:
```console
❱❱❱ python myTest.py -path=<path_to_saved_model> 
```

## OOV Test
If you want to train model under the OOV Test (please refer to our paper for details), you can add `-ot` to the function call.

## Acknowledgement

**Global-to-local Memory Pointer Networks for Task-Oriented Dialogue**. [Chien-Sheng Wu](https://jasonwu0731.github.io/), [Richard Socher](https://www.socher.org/), [Caiming Xiong](http://www.stat.ucla.edu/~caiming/). ***ICLR 2019***. [[PDF]](https://arxiv.org/abs/1901.04713) [[Open Reivew]](https://openreview.net/forum?id=ryxnHhRqFm) [[Code]](https://github.com/jasonwu0731/GLMP)

>   We are highly grateful for the public code of GLMP!
