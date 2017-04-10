from typ import Typ, TypSymbol, TypVar, TypTerm


def mgu(t1: Typ, t2: Typ):
    # TODO: possibly optimize by removing the non-local
    # variables

    table = {}
    fail = None
    agenda = [(t1, t2)]

    def mgu_process_var(var, typ):
        nonlocal fail, table, agenda
        if not isinstance(typ, Typ):
            fail = "Not a Typ: '%s'" % (typ)
            return
        if typ.contains_var(var):
            fail = "Occur check fail: '%s' in '%s'" % (var, typ)
            return
        if var == typ:
            return

        for key, val in table.items():
            table[key] = val.apply_mini_sub(var, typ)

        agenda = [(f.apply_mini_sub(var, typ),
                   s.apply_mini_sub(var, typ)) for f, s in agenda]

        table[var] = typ

    def mgu_process_term(term1, term2):
        nonlocal fail, agenda

        if len(term1.arguments) != len(term2.arguments):
            fail = "Term length mismatch: '%s' and '%s'" % (term1, term2)
            return

        agenda.extend(zip(term1.arguments, term2.arguments))

    def mgu_process(a1, a2):
        nonlocal fail
        if a1 == a2:
            return

        if isinstance(a1, TypSymbol) and isinstance(a2, TypSymbol):
            fail = "Symbols mismatch: '%s' and '%s'" % (a1, a2)
            return

        if isinstance(a1, TypVar):
            mgu_process_var(a1, a2)
            return

        if isinstance(a2, TypVar):
            mgu_process_var(a2, a1)
            return

        if isinstance(a1, TypTerm) and isinstance(a2, TypTerm):
            mgu_process_term(a1, a2)
            return

        if ((isinstance(a1, TypTerm) and isinstance(a2, TypSymbol)) or
                (isinstance(a2, TypTerm) and isinstance(a1, TypSymbol))):
            fail = "Kind mismatch: '%s' and '%s'" % (a1, a2)
            return

        fail = "Not Typs: '%s' and '%s'" % (a1, a2)

    while agenda and not fail:
        a1, a2 = agenda.pop()
        mgu_process(a1, a2)

    return Sub(table, fail)


class Sub:
    def __init__(self, table={}, fail_msg=None):
        self.table = table
        self.fail_msg = fail_msg

    def __eq__(self, other):
        if not self.fail_msg:
            return self.table == other.table

        raise RuntimeError("Comparing two failed substitutions.")

    def __repr__(self):
        if not self.fail_msg:
            return "Sub(%s)" % (self.table)
        else:
            return "Sub(fail_msg='%s')" % (self.fail_msg)

    def is_failed(self):
        return bool(self.fail_msg)


if __name__ == "__main__":
    pass
