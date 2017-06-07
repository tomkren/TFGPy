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

            print("%s @ %d\tmax_fitness=%.3f\t%s" % (repr(self), len(self.d), self.max_val, self.max_key))

    def __len__(self):
        return len(self.d)

    def __repr__(self):
        return "FitnessCache<%s>" % id(self)