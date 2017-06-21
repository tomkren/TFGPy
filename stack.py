import copy
import random


class Stack:
    def __init__(self, symbols_d):
        self.symbols_d = symbols_d

    def count_missing(self, stack):
        missing = 1
        non_terminals = 0
        terminals = 0

        for symbol in stack:
            arity, _ = self.symbols_d.get(symbol, (0, None))
            missing += arity - 1
            if arity:
                non_terminals += 1
            else:
                terminals += 1
        return missing, terminals, non_terminals

    def successor_symbols(self, stack, limit, count=None):
        missing, terminals, non_terminals = count if count is not None else self.count_missing(stack)

        if not missing:
            return []
        return [s for s, (arity, _) in self.symbols_d.items()
                if terminals + non_terminals + missing + arity <= limit]

    def finish(self, stack, limit):
        missing, terminals, non_terminals = self.count_missing(stack)

        assert missing >= 0
        if not missing:
            return stack
        stack = copy.copy(stack)

        while missing > 0:
            ssym = self.successor_symbols(stack, limit, (missing, terminals, non_terminals))
            assert ssym
            ns = random.choice(ssym)

            arity, _ = self.symbols_d[ns]
            if not arity:
                missing -= 1
                terminals += 1
            else:
                missing += arity - 1
                non_terminals += 1

            stack.append(ns)

        return stack

    def is_finished(self, stack):
        nonfinished = self.count_missing(stack)[0]
        return nonfinished == 0

    def successors(self, stack, limit):
        count = self.count_missing(stack)
        ssym = self.successor_symbols(stack, limit, count)

        if not ssym:
            return []

        return [stack + [s] for s in ssym]

    def eval_stack(self, stack, val_d):
        s = copy.copy(stack)
        v = []

        while len(s):
            # print(s, v)
            symbol = s.pop()
            if symbol in self.symbols_d:
                arity, fn = self.symbols_d[symbol]
                if arity == 0:
                    assert symbol in val_d
                    v.append(val_d[symbol])
                else:
                    args = reversed(v[-arity:])
                    v[-arity:] = []
                    v.append(fn(*args))
            else:
                v.append(symbol)

        assert len(v) == 1 and not s
        # print(s, v)
        return v[0]
