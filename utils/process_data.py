# from tqdm import tqdm
import gensim
import os
from collections import Counter, defaultdict


from utils.huffman import HuffmanTree
from utils.read_bible import load_bible, split_bible_on_chapters

UNK = '<UNK>'


def read_data(path):
    def _read_file(filepath):
        data = []
        with open(filepath) as f:
            for l in f:
                data.append(l)
        return data

    if os.path.isfile(path):
        data = _read_file(path)
        return data
    elif os.path.isdir(path):
        data = []
        for file in os.listdir(path):
            fullpath = os.path.join(file)
            _d = _read_file(fullpath)
            _d = ' '.join(_d)
            data.append(_d)
        return data
    else:
        raise FileNotFoundError(f'Non-correct path: {path} is given!')


def tokenize(text_lines):
    # for i, line in tqdm(enumerate(text_array),
    #                     total=len(text_array),
    #                     desc='read_corpus: '):
    for i, line in enumerate(text_lines):
        yield gensim.utils.simple_preprocess(line)


def get_vocab_counter(corpus):
    vocab_counter = Counter()
    for sentence in corpus:
        for word in sentence:
            vocab_counter[word] += 1
    return vocab_counter


def replace_words_by_inxs(corpus, hash, flat_input=False, flat_output=False,
                            use_unk=False):
    if flat_input is False:
        new_corpus = []
        for sentence in corpus:
            if not flat_output: new_sentence = []
            for word in sentence:
                if use_unk is True:
                    inx = hash.get(word, hash.get(UNK))
                else:
                    inx = hash[word]
                if not flat_output: new_sentence.append(inx)
                else:
                    new_corpus.append(inx)
            if not flat_output:
                new_corpus.append(new_sentence)
    else:
        new_corpus = []
        # for w in tqdm(corpus, desc='replace_words_by_tokens: '):
        for w in corpus:
            if use_unk is True:
                inx = hash.get(w, hash.get(UNK))
            else:
                inx = hash[w]
            new_corpus.append(inx)
    return new_corpus


def downsample(corpus, words_set, factor):
    new_corpus = []
    counts = defaultdict(int)
    for doc in corpus:
        new_doc = []
        for w in doc:
            if w in words_set:
                counts[w] += 1
                if counts[w] == factor:
                    new_doc.append(w)
                    counts[w] = 0
            else:
                new_doc.append(w)
        new_corpus.append(new_doc)
    return new_corpus


def process_data(datapath='./bible.txt',
                 threshold_count=5,
                 pipeline='hier_softmax',
                 downsampling_params=None,
                 debug=False):
    """

    :param datapath: './bible.txt' by default /
        any .txt file or folder with .txt files is supported;
    :param threshold_count: replace words with frequency < threshold_count
        by UNK
    :param pipeline: "hier_softmax"/"neg_sampling"
    :param downsampling_params: None/(topK:int, factor: int);
        if not None, drop topK most frequent words by factor
    :param debug:
    :return: in case of "hier_softmax" pipeline returns:
        huffman_corpus, node_inxs, turns_inxs, leaves_hash_inversed,\
                vocab_size, extended_vocab_size, vocab, huffman_tree
            in case of "negative_sampling" pipeline returns:
        corpus, words_hash_inversed, vocab_size

    """
    if datapath == './bible.txt':
        raw_bible = load_bible(datapath)
        sentences = split_bible_on_chapters(raw_bible)
    else:
        sentences = read_data(datapath)

    corpus = list(tokenize(sentences))  # only for small corpus
    vocab_counter = get_vocab_counter(corpus)

    # do downsampling
    if downsampling_params is not None:
        topk, factor = downsampling_params
        words_set = sorted(list(vocab_counter.items()),
                           key=lambda x: x[1], reverse=True)[:topk]
        corpus = downsample(corpus, words_set, factor)
        vocab_counter = get_vocab_counter(corpus)

    # filter vocab
    vocab = set((w for w, c in vocab_counter.items() if
                 c >= threshold_count))
    vocab.add(UNK)
    new_vocab_counter = defaultdict(int)
    for w, c in vocab_counter.items():
        if w in vocab:
            new_vocab_counter[w] = c
        else:
            new_vocab_counter[UNK] += c
    vocab_counter = new_vocab_counter
    del new_vocab_counter
    vocab_size = len(vocab)

    if pipeline == 'hier_softmax':
        # huffman tree
        huffman_tree = HuffmanTree(vocab_counter.items())
        leaves_hash, leaves_hash_inversed = huffman_tree.get_leaves_hash()
        node_inxs, turns_inxs = huffman_tree.gather_pathes()
        extended_vocab_size = len(huffman_tree.nodes_enumerated)
        huffman_corpus = replace_words_by_inxs(corpus,
                                               leaves_hash,
                                               flat_input=False,
                                               flat_output=True,
                                               use_unk=True)
        if debug:
            return huffman_corpus, node_inxs, turns_inxs, leaves_hash_inversed,\
                vocab_size, extended_vocab_size, vocab, huffman_tree
        else:
            return huffman_corpus, node_inxs, turns_inxs, \
                   leaves_hash_inversed, vocab_size, extended_vocab_size
    elif pipeline == 'neg_sampling':
        words_hash = {w:i for i,w in enumerate(vocab)}
        words_hash_inversed = {i:w for w,i in words_hash.items()}
        corpus = replace_words_by_inxs(corpus,
                                       words_hash,
                                       flat_input=False,
                                       flat_output=True,
                                       use_unk=True)
        return corpus, words_hash_inversed, vocab_size
    else:
        raise NotImplementedError(f"Doesn't support the 'pipeline': {pipeline}")

