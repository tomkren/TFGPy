class FitnessCache:
    def __init__(self):
        self.d = {}
        self.max_val, self.max_key = None, None

    def update(self, key, value):
        assert key not in self.d
        self.d[key] = value
        if self.max_key is None or self.max_val < value:
            self.max_key = key
            self.max_val = value
            self.print_self()

    def print_self(self, label=''):
        self.print_val(self.max_val, self.max_key, label)

    def print_val(self, value, key, label=''):
        print("%s @ %s %d\tmax_fitness=%E\t%s" % (repr(self), label, len(self.d), value, key), flush=True)

    def __len__(self):
        return len(self.d)

    def __repr__(self):
        return "FitnessCache<%s>" % id(self)
