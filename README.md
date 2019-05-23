- Train word2vec by negative sampling approach on default dataset (data/bible.txt): <br>
```
python3 train.py -c=configs/conf_neg_sampling.json -f=../test_word2vec
```

- by hierarchical softmax technique: <br>
```
python3 train.py -c=configs/conf_hier_softmax.json -f=../test_ns_word2vec
```
