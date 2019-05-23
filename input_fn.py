import numpy as np

import torch
from torch.utils.data import Dataset


class CBOWDataSet(Dataset):
    def __init__(self, corpus,
                 pipeline='hier_softmax',
                 nodes_index=None,
                 turns_index=None,
                 vocab_size=None,
                 neg_samples=None,
                 max_path_len=17,
                 window_size=6,
                 device=None,
                 skip_target=False,
                 dtype=torch.float32):
        """

        :param corpus: the flat list of tokens
        :param pipeline: 'hier_softmax'/'neg_sampling'
        params for 'hierarchical softmax' pipeline:
            :param nodes_index: index of nodes from leaf parent to the root
            :param turns_index: the list of 1/-1 indices:
                1 — the leaf is the left child of corresponding node
                -1 — the leaf is the right child
            :param vocab_size: is used for padding
            :param max_path_len: length of the longest path from word (leaf)
            to the root

        params for 'negative sampling' pipeline:
            :param neg_samples: the number of negative samples

        :param window_size: word context size
        :param device: cuda:0/cuda:1/cpu
        :param dtype:  torch float type
        """
        self.window_size = window_size
        self.step = window_size // 2
        self.left_step = self.step
        self.right_step = window_size - self.step
        self.corpus = corpus[-self.left_step:] + corpus + \
            corpus[:self.right_step]

        self.device = device
        self.dtype = dtype

        self.pipeline = pipeline
        if self.pipeline == 'hier_softmax':
            self.nodes_index = nodes_index
            self.max_path_len = max_path_len
            self.turns_index = turns_index
            self.vocab_size = vocab_size
            self.skip_target = skip_target
        elif self.pipeline == 'neg_sampling':
            self.np_corpus = np.array(self.corpus)
            self.neg_samples = neg_samples
        else:
            raise NotImplementedError(
                f'Pipeline for "pipeline": {self.pipeline}')

    def __len__(self):
        return len(self.corpus) - self.window_size

    def __getitem__(self, item):
        if self.pipeline == 'hier_softmax':
            return self.__h_getitem(item)
        elif self.pipeline == 'neg_sampling':
            return self.__n_getitem(item)
        else:
            raise NotImplementedError(
                f'__getitem__ for pipeline: {self.pipeline}')

    def __h_getitem(self, i):
        """
        Hierarchical softmax pipepline
        :param i: item index
        :return: torch tensors:
            context, target, nodes, mask, turns_coeffs
        """
        i += self.left_step
        target = self.corpus[i]
        context = self.corpus[(i - self.left_step):i]
        context += self.corpus[(i + 1):(i + self.right_step + 1)]
        try:
            assert len(context) == self.window_size
        except AssertionError:
            raise Exception(
                'Context size is not valid: context - '
                '{0} has size - {1}; window_size - {2}'
                .format(context, len(context), self.window_size)
            )

        nodes = self.nodes_index[target]
        nodes_len = len(nodes)
        mask = np.zeros(self.max_path_len)
        mask[:nodes_len] = 1

        pad_len = self.max_path_len - nodes_len
        nodes = np.concatenate([nodes, np.ones(pad_len) * self.vocab_size])
        #         nodes = np.concatenate([nodes, np.ones(pad_len) * -1])
        nodes = torch.tensor(nodes, dtype=torch.long, device=self.device)

        turns_coeffs = self.turns_index.get(target)
        turns_coeffs = np.concatenate([turns_coeffs, np.zeros(pad_len)])
        turns_coeffs = torch.tensor(turns_coeffs, dtype=self.dtype,
                                    device=self.device)
        mask = torch.tensor(mask, dtype=self.dtype, device=self.device)

        context = torch.tensor(context, dtype=torch.long, device=self.device)
        target = torch.tensor(target, dtype=torch.long, device=self.device)

        if self.skip_target is False:
            return context, target, nodes, mask, turns_coeffs
        else:
            return context, nodes, mask, turns_coeffs

    def __n_getitem(self, i):
        """
        Negative sampling pipeline
        :param i: item index
        :return: torch tensors:
            context, target, neg_samples
        """
        i += self.left_step
        target = self.corpus[i]
        context = self.corpus[(i - self.left_step):i]
        context += self.corpus[(i + 1):(i + self.right_step + 1)]
        try:
            assert len(context) == self.window_size
        except AssertionError:
            raise Exception(
                'Context size is not valid: context - '
                '{0} has size - {1}; window_size - {2}'
                    .format(context, len(context), self.window_size)
            )
        context = torch.tensor(context, dtype=torch.long, device=self.device)
        target = torch.tensor(target, dtype=torch.long, device=self.device)

        return context, target
