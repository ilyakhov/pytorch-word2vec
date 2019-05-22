from collections import (Counter, defaultdict)

from utils.huffman import HuffmanTree
from utils.process_data import (replace_words_by_inxs, downsample, process_data)


def test_huffman():
    test_vocab = ['foo']*10 + ['bar']*5
    huffman_tree = HuffmanTree(Counter(test_vocab).items())
    print(huffman_tree)
    node_inxs, turn_inxs = huffman_tree.gather_pathes()
    assert huffman_tree.root.weight == 15
    assert huffman_tree.get_leaves()[0].word == 'bar'
    assert huffman_tree.get_leaves()[1].word == 'foo'
    assert tuple(node_inxs[1]) == (2,)
    assert tuple(node_inxs[0]) == (2,)
    assert tuple(turn_inxs[1]) == (1,)
    assert tuple(turn_inxs[0]) == (-1,)

    test_vocab = ['foo'] * 5 + ['bar'] * 10
    huffman_tree = HuffmanTree(Counter(test_vocab).items())
    print(huffman_tree)
    node_inxs, turn_inxs = huffman_tree.gather_pathes()
    assert huffman_tree.root.weight == 15
    assert huffman_tree.get_leaves()[1].word == 'bar'
    assert huffman_tree.get_leaves()[0].word == 'foo'
    assert tuple(node_inxs[0]) == (2,)
    assert tuple(node_inxs[1]) == (2,)
    assert tuple(turn_inxs[0]) == (1,)
    assert tuple(turn_inxs[1]) == (-1,)

    test_vocab = ['foo'] * 5 + ['bar'] * 10 + ['zoo'] * 5
    huffman_tree = HuffmanTree(Counter(test_vocab).items())
    print(huffman_tree)
    assert tuple(n.word for n in huffman_tree.get_leaves()) == \
        ('bar', 'foo', 'zoo')
    assert huffman_tree.root.weight == 20
    node_inxs, turn_inxs = huffman_tree.gather_pathes()
    assert tuple(node_inxs[1]) == (4,)
    assert tuple(node_inxs[0]) == (3, 4)
    assert tuple(node_inxs[2]) == (3, 4)
    assert tuple(turn_inxs[1]) == (1,)
    assert tuple(turn_inxs[0]) == (1, -1)
    assert tuple(turn_inxs[2]) == (-1, -1)

    test_vocab = ['foo'] * 1 + ['bar'] * 2 + ['zoo'] * 1
    huffman_tree = HuffmanTree(Counter(test_vocab).items())
    print(huffman_tree)
    assert huffman_tree.root.weight == 4

    test_vocab = ['foo'] * 1 + ['foo'] * 2 + ['foo'] * 1 # one word -> the root?
    huffman_tree = HuffmanTree(Counter(test_vocab).items())
    print(huffman_tree)
    assert huffman_tree.root.weight == 4


def test_process_data():
    import numpy as np

    def flat_input_tests(corpus, hash):
        corpus_inxs = defaultdict(list)
        for i, w in enumerate(corpus):
            corpus_inxs[w].append(i)
        new_corpus = replace_words_by_inxs(corpus, hash, flat_input=True)
        new_corpus = np.array(new_corpus)

        for w in corpus_inxs:
            inxs = np.where(new_corpus == hash[w])[0]
            assert tuple(inxs) == tuple(corpus_inxs[w])

    corpus = ['the', 'foo', 'bar', 'the', 'the', 'the',
              'foo', 'bar', 'the', 'foo', 'the', 'foo']
    hash = {'the': 0, 'foo': 1, 'bar': 2}
    flat_input_tests(corpus, hash)

    corpus = ['the', 'foo', 'bar', 'monthy', 'python', 'the', 'flying',
              'circus', 'the', 'holy', 'grail', 'guido', 'van', 'rossum']
    hash = {'the': 0, 'foo': 1, 'bar': 2, 'monthy': 3, 'python': 4,
            'flying': 5, 'circus': 6, 'holy': 7, 'grail': 8, 'guido': 9,
            'van': 10, 'rossum': 11}
    flat_input_tests(corpus, hash)

    corpus = [['the', 'foo', 'bar', 'monthy'],
              ['python', 'the', 'flying', 'circus', 'the', 'holy',
               'grail', 'guido', 'van', 'rossum']]
    new_corpus = replace_words_by_inxs(corpus, hash,
                                       flat_input=False,
                                       flat_output=False)
    assert tuple(new_corpus[0]) == (0, 1, 2, 3)
    assert tuple(new_corpus[1]) == (4, 0, 5, 6, 0, 7, 8, 9, 10, 11)

    corpus = [['the', 'foo', 'bar', 'monthy'],
              ['python', 'the', 'flying', 'circus', 'the', 'holy',
               'grail', 'guido', 'van', 'rossum']]
    new_corpus = replace_words_by_inxs(corpus, hash,
                                       flat_input=False,
                                       flat_output=True)
    assert tuple(new_corpus) == (0, 1, 2, 3, 4, 0, 5, 6, 0, 7, 8, 9, 10, 11)

    # process_data tests on bible_sample.txt
    huffman_corpus, node_inxs, turns_inxs, leaves_hash_inversed, \
        vocab_size, extended_vocab_size, vocab, huffman_tree = \
        process_data('./bible_sample_1.txt', 1, debug=True)
    print(vocab_size)
    print(extended_vocab_size)
    print(leaves_hash_inversed)

    print(huffman_corpus)
    assert tuple(huffman_corpus) == (0, 0, 0, 1, 0, 0, 1, 2, 2, 0, 0, 2, 1)
    print(vocab)
    assert vocab == {'foo', '<UNK>', 'bar', 'the'}
    print(node_inxs)
    print(leaves_hash_inversed[0], node_inxs[0])
    assert tuple(node_inxs[0]) == (4,)
    print(leaves_hash_inversed[1], node_inxs[1])
    assert tuple(node_inxs[1]) == (3, 4)
    print(leaves_hash_inversed[2], node_inxs[2])
    assert tuple(node_inxs[2]) == (3, 4)

    print(leaves_hash_inversed[0], turns_inxs[0])
    assert tuple(turns_inxs[0]) == (-1,)
    print(leaves_hash_inversed[1], turns_inxs[1])
    assert tuple(turns_inxs[1]) == (1, 1)
    print(leaves_hash_inversed[2], turns_inxs[2])
    assert tuple(turns_inxs[2]) == (-1, 1)
    print(huffman_tree)

    # test downsample
    c = [['foo', 'moon', 'bar'],
         ['foo', 'foo', 'foo', 'foo', 'bar', 'sun'],
         ['bar', 'bar', 'bar', 'zoo']]

    assert tuple(tuple(x) for x in downsample(c, {'foo', 'bar'}, 1)) != \
        (('moon',), ('sun',), ('zoo',))
    assert tuple(tuple(x) for x in downsample(c, {'foo', 'bar'}, 10)) ==\
        (('moon',), ('sun',), ('zoo',))
    assert tuple(tuple(x) for x in downsample(c, {'foo', 'bar'}, 100)) == \
        (('moon',), ('sun',), ('zoo',))
    assert tuple(tuple(x) for x in downsample(c, {'foo', 'bar'}, 5)) == \
        (('moon',), ('foo', 'sun'), ('bar', 'zoo'))


if __name__ == '__main__':
    test_huffman()
    test_process_data()
