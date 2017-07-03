from app_tree import AppTree, Leaf

#
#   Statistics on subtrees
#       - to be used for heuristics during search / smart playouts
#
from tree_node import UFTNode


class RunningStat:
    def __init__(self, count=0, sum=0):
        self.count = count
        self.sum = sum
        self.biggest = None
        self.smallest = None

    def add(self, value):
        self.count += 1
        self.sum += value
        if self.biggest is None or value > self.biggest:
            self.biggest = value
        if self.smallest is None or value < self.smallest:
            self.smallest = value

    def avg(self):
        if not self.count:
            raise ValueError
        return self.sum / self.count


class Stats:
    def __init__(self):
        self.total = RunningStat()
        self.by_tree = {}

    def update(self, tree, score):
        self.total.add(score)
        self.by_tree.setdefault(tree, RunningStat()).add(score)


class TreeStats:
    def __init__(self):
        self.typ2size2stats = {}

    def update(self, node, score):
        if not isinstance(node, UFTNode):
            return
        root = node.uf_tree
        assert isinstance(root, AppTree)

        def update_one(tree):
            assert tree.typ is not None
            assert tree.is_finished()
            counts = tree.count_nodes()
            k = counts[Leaf]

            size2stats = self.typ2size2stats.setdefault(tree.typ, {})
            size2stats.setdefault(k, Stats()).update(tree, score)

        root.map_reduce(update_one, (lambda *args: None))

    def pretty_str(self):
        l = []
        for typ, size2stats in self.typ2size2stats.items():
            l.append('=' * 10 + str(typ) + '=' * 10)
            for k, stats in sorted(size2stats.items()):
                t, rs = max(stats.by_tree.items(), key=(
                    lambda t: t[1].avg()
                    #lambda t: t[1].biggest
                ))
                l.append("k=%d %d %.3f %s" % (k, rs.count, rs.avg(), t))
        return '\n'.join(l)