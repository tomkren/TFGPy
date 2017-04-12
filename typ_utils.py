from typ import TypSymbol, InfixBinTerm

T_ARROW = TypSymbol('->')


def mk_fun_type(t_from, t_to):
    return InfixBinTerm(T_ARROW, t_from, t_to)
