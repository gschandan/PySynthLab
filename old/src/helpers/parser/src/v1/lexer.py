import re

import ply.lex

from ..base.lexer import SygusLexerBase


# noinspection PyPep8Naming
class SygusV1Lexer(SygusLexerBase):
    tokens = SygusLexerBase.tokens
    reserved = SygusLexerBase.reserved

    reserved['declare-fun'] = 'TK_DECLARE_FUN'
    reserved['set-options'] = 'TK_SET_OPTIONS'
    reserved['declare-primed-var'] = 'TK_DECLARE_PRIMED_VAR'
    reserved['BitVec'] = 'TK_BV'
    reserved['Int'] = 'TK_INT'
    reserved['Bool'] = 'TK_BOOL'
    reserved['Real'] = 'TK_REAL'
    reserved['String'] = 'TK_STRING'
    reserved['Enum'] = 'TK_ENUM'
    reserved['Array'] = 'TK_ARRAY'

    for token_name in set(reserved.values()):
        if token_name not in tokens:
            tokens.append(token_name)

    if 'TK_DOUBLE_COLON' not in tokens:
        tokens.append('TK_DOUBLE_COLON')

    @ply.lex.TOKEN(SygusLexerBase._symbol)
    def t_TK_SYMBOL(self, t):
        t.type = self.reserved.get(t.value, 'TK_SYMBOL')
        if t.type == 'TK_SYMBOL' and re.fullmatch(r'-\d+', t.value):
            t.value = int(t.value)
            t.type = 'TK_NUMERAL'
        return t

    def __init__(self):
        super().__init__()
