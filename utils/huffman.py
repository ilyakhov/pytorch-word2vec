import heapq
from copy import copy, deepcopy
from collections import defaultdict


class Node:
    def __init__(self, parent, left, right, weight, word=None):
        self.parent = parent
        self.left = left
        self.right = right
        self.weight = weight
        self.word = word
        self.nodes_enumerated = None
        self.nodes_enumerated_inversed = None

    def __gt__(self, node2):
        return self.weight > node2.weight


class HuffmanTree:
    def __init__(self, word_frequencies=None):
        self.root = Node(None, None, None, -1)
        if word_frequencies is not None:
            self.__build_tree(word_frequencies)

    def __call__(self, word_frequencies):
        return self.__build_tree(word_frequencies)

    def __build_tree(self, word_frequencies):
        # define leaves
        leaves = []
        nodes_enumerated = dict()

        if len(word_frequencies) == 1:
            self.root.word = list(word_frequencies)[0][0]
            self.root.weight = list(word_frequencies)[0][1]
            nodes_enumerated[self.root] = 0
        else:
            for i, (word, freq) in enumerate(word_frequencies):
                node_to_push = Node(None, None, None, freq, word)
                heapq.heappush(leaves, node_to_push)

                # the first N enumerated nodes is for leaves
                nodes_enumerated[node_to_push] = i

            # greedy procedure
            # 1. "stack" of unused nodes: build new node with minimum weight
            # 2. stop it when nodes_stack is empty
            nodes_stack = leaves
            node_iter = max(nodes_enumerated.values())
            while nodes_stack:
                if len(nodes_stack) == 2:
                    left = heapq.heappop(nodes_stack)
                    right = heapq.heappop(nodes_stack)
                    self.root.left = left
                    self.root.right = right
                    self.root.weight = left.weight + right.weight
                    left.parent = self.root
                    right.parent = self.root

                    node_iter += 1
                    nodes_enumerated[self.root] = node_iter
                elif len(nodes_stack) == 1:
                    left = heapq.heappop(nodes_stack)
                    self.root.left = left
                    self.root.weight = left.weight
                    left.parent = self.root

                    node_iter += 1
                    nodes_enumerated[self.root] = node_iter
                else:
                    left = heapq.heappop(nodes_stack)
                    right = heapq.heappop(nodes_stack)
                    new_node = Node(None, left, right,
                                    left.weight + right.weight)
                    left.parent = new_node
                    right.parent = new_node
                    heapq.heappush(nodes_stack, new_node)

                    node_iter += 1
                    nodes_enumerated[new_node] = node_iter

        assert self.root.weight != -1
        self.nodes_enumerated = nodes_enumerated
        self.nodes_enumerated_inversed = {i: n for n, i in
                                          self.nodes_enumerated.items()}
        return self

    def get_leaves_hash(self):
        leaves = self.get_leaves()
        leaves_hash = {l.word: self.nodes_enumerated[l] for l in leaves}
        leaves_hash_inversed = {l: w for w, l in leaves_hash.items()}
        return leaves_hash, leaves_hash_inversed

    def gather_pathes(self):
        """
        Collects all pathes from the root to leaves to get inverted pathes
        in the form of indices corresponded to each nodes in the path:
         --a) node_inxs: [leaveN_inx, node_K, node_K+1, ..., root_inx]-- (delete it!)
         a) node_inxs: [node_K, node_K+1, ..., root_inx]
        and indexes of turns 1 for the left child; -1 for the right turn:
         b) turns_inxs: [1, -1, ... -1] — the array with length (len(nodes_list) - 1)
        """
        leaves = self.get_leaves()
        node_inxs = defaultdict(list)
        turns_inxs = defaultdict(list)
        root_node_inx = self.nodes_enumerated[self.root]
        for leaf in leaves:
            current_node = leaf
            leaf_inx = self.nodes_enumerated[current_node]
            current_node_inxs = []
            current_turns_inxs = []
            while current_node != self.root:
                node_inx = self.nodes_enumerated[current_node]
                current_node_inxs.append(node_inx)

                parent = current_node.parent
                if parent.left == current_node:
                    inx = 1
                elif parent.right == current_node:
                    inx = -1
                current_turns_inxs.append(inx)

                current_node = parent

            current_node_inxs.append(root_node_inx)
            node_inxs[leaf_inx] = current_node_inxs[
                                  1:]  # drop itself. TODO: fix this crook!
            turns_inxs[leaf_inx] = current_turns_inxs
        return node_inxs, turns_inxs

    # helpers
    def get_leaves(self):
        def _recursive_traverse(node, leaves):
            if node.left is None and node.right is None:
                leaves.append(node)
            else:
                if node.left:
                    leaves = _recursive_traverse(node.left, leaves)
                if node.right:
                    leaves = _recursive_traverse(node.right, leaves)
            return leaves

        leaves = []
        return _recursive_traverse(self.root, leaves)

    def __repr__(self):
        nodes_to_print = [self.root]
        buffer = ''
        total_print_message = ''
        while nodes_to_print:
            new_nodes_to_print = []
            print_message = buffer
            for node in nodes_to_print:
                print_message += 'i{}:w{}:{}\t'.format(
                    self.nodes_enumerated[node], node.weight, node.word
                )
                if node.left is not None:
                    new_nodes_to_print.append(node.left)
                else:
                    buffer += ' '
                if node.right is not None:
                    new_nodes_to_print.append(node.right)
                else:
                    buffer += ' '
            total_print_message += print_message + '\n'
            nodes_to_print = new_nodes_to_print[:]
            del new_nodes_to_print
        return total_print_message