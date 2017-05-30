class TNode:
    def __str__(self):
        return "TNode<>"

    def is_finished(self):
        raise NotImplementedError


class UFTNode(TNode):
    def __init__(self, uf_tree, k):
        self.uf_tree = uf_tree
        self.k = k

    def __str__(self):
        return "UFTNode<k=%d, %s>" % (self.k, self.uf_tree)

    def is_finished(self):
        return self.uf_tree.is_finished()


class ChooseKTNode(TNode):
    def __init__(self, uf_tree, max_k):
        self.uf_tree = uf_tree
        self.max_k = max_k

    def __str__(self):
        return "ChooseKTNode<max_k=%d, %s>" % (self.max_k, self.uf_tree)

    def is_finished(self):
        return self.uf_tree.is_finished()


class StackNode(TNode):
    def __init__(self, stack):
        self.stack = stack

    def __str__(self):
        return "StackNode<%s>" % (self.stack)
