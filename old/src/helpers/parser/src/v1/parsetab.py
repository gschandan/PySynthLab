# parsetab.py
# This file is automatically generated. Do not edit.
# pylint: disable=W,C,R
_tabversion = '3.10'

_lr_method = 'LALR'

_lr_signature = 'programTK_ARRAY TK_BIN_CONST TK_BOOL TK_BOOL_CONST TK_BV TK_CHECK_SYNTH TK_COLON TK_CONSTANT TK_CONSTRAINT TK_DECIMAL TK_DECLARE_DATATYPE TK_DECLARE_DATATYPES TK_DECLARE_FUN TK_DECLARE_PRIMED_VAR TK_DECLARE_SORT TK_DECLARE_VAR TK_DEFINE_FUN TK_DEFINE_SORT TK_DOUBLE_COLON TK_ENUM TK_EXISTS TK_FORALL TK_HEX_CONST TK_INT TK_INV_CONSTRAINT TK_LET TK_LPAREN TK_NUMERAL TK_PAR TK_REAL TK_RPAREN TK_SET_FEATURE TK_SET_INFO TK_SET_LOGIC TK_SET_OPTION TK_SET_OPTIONS TK_STRING TK_STRING_LITERAL TK_SYMBOL TK_SYNTH_FUN TK_SYNTH_INV TK_UNDERSCORE TK_VARIABLEcommand : fun_def_command\n                   | fun_decl_command\n                   | synth_fun_command\n                   | synth_inv_command\n                   | check_synth_command\n                   | constraint_command\n                   | inv_constraint_command\n                   | sort_def_command\n                   | set_opts_command\n                   | var_decl_command\n                   | primed_var_decl_commandprogram : set_logic_command command_plus\n                   | command_plusfun_def_command : TK_LPAREN TK_DEFINE_FUN symbol arg_list sort_expr term TK_RPARENfun_decl_command : TK_LPAREN TK_DECLARE_FUN symbol TK_LPAREN sort_star TK_RPAREN sort_expr TK_RPARENcommand_plus : command_plus command\n                        | commandsynth_fun_command : TK_LPAREN TK_SYNTH_FUN symbol arg_list sort_expr TK_LPAREN nonterminal_def_plus TK_RPAREN TK_RPAREN\n                             | TK_LPAREN TK_SYNTH_FUN symbol arg_list sort_expr TK_RPARENcheck_synth_command : TK_LPAREN TK_CHECK_SYNTH TK_RPARENconstraint_command : TK_LPAREN TK_CONSTRAINT term TK_RPARENvar_decl_command : TK_LPAREN TK_DECLARE_VAR TK_SYMBOL sort_expr TK_RPARENsynth_inv_command : TK_LPAREN TK_SYNTH_INV symbol arg_list TK_LPAREN nonterminal_def_plus TK_RPAREN TK_RPAREN\n                            | TK_LPAREN TK_SYNTH_INV symbol arg_list TK_RPARENinv_constraint_command : TK_LPAREN TK_INV_CONSTRAINT TK_SYMBOL TK_SYMBOL TK_SYMBOL TK_SYMBOL TK_RPARENset_logic_command : TK_LPAREN TK_SET_LOGIC TK_SYMBOL TK_RPARENsort_def_command : TK_LPAREN TK_DEFINE_SORT symbol sort_expr TK_RPARENliteral : TK_NUMERALliteral : TK_BOOL_CONSTset_opts_command : TK_LPAREN TK_SET_OPTIONS option_list TK_RPARENliteral : TK_DECIMALprimed_var_decl_command : TK_LPAREN TK_DECLARE_PRIMED_VAR symbol sort_expr TK_RPARENliteral : TK_HEX_CONSTliteral : TK_BIN_CONSTsort_expr : TK_LPAREN TK_BV TK_NUMERAL TK_RPARENliteral : TK_STRING_LITERALsort_expr : TK_INTarg_list : TK_LPAREN sorted_symbol_star TK_RPARENsorted_symbol_star : sorted_symbol_star sorted_symbol\n                              | sort_expr : TK_BOOLsort_expr : TK_REALnonempty_arg_list : TK_LPAREN sorted_symbol_plus TK_RPARENsorted_symbol_plus : sorted_symbol_plus sorted_symbol\n                              | sorted_symbolsort_expr : TK_STRINGsort_expr : TK_LPAREN TK_ENUM enum_constructor_list TK_RPARENsorted_symbol : TK_LPAREN TK_SYMBOL sort_expr TK_RPARENsort_expr : TK_LPAREN TK_ARRAY sort_expr sort_expr TK_RPARENsort_expr : TK_SYMBOLsymbol : TK_SYMBOLenum_constructor_list : TK_LPAREN symbol_plus TK_RPARENsymbol_plus : symbol_plus symbol\n                       | symbolterm : literal\n                | TK_SYMBOL\n                | TK_LPAREN symbol term_star TK_RPAREN\n                | TK_LPAREN TK_EXISTS nonempty_arg_list term TK_RPAREN\n                | TK_LPAREN TK_FORALL nonempty_arg_list term TK_RPAREN\n                | let_termlet_term : TK_LPAREN TK_LET TK_LPAREN binding_plus TK_RPAREN term TK_RPARENbinding_plus : binding_plus binding\n                        | bindingbinding : TK_LPAREN symbol sort_expr term TK_RPARENterm_star : term_star term\n                     | literal : enum_constenum_const : TK_SYMBOL TK_DOUBLE_COLON TK_SYMBOLsort_star : sort_star sort_expr\n                     | option_list : TK_LPAREN symbol_pair_plus TK_RPARENsymbol_pair_plus : symbol_pair_plus symbol_pair\n                            | symbol_pairsymbol_pair : TK_LPAREN symbol TK_STRING_LITERAL TK_RPARENnonterminal_def_plus : nonterminal_def_plus nonterminal_def\n                                | nonterminal_defnonterminal_def : TK_LPAREN symbol sort_expr TK_LPAREN grammar_term_plus TK_RPAREN TK_RPARENgrammar_term_plus : grammar_term_plus grammar_term\n                             | grammar_termgrammar_term : TK_SYMBOLgrammar_term : literalgrammar_term : TK_LPAREN symbol grammar_term_star TK_RPARENgrammar_term : TK_LPAREN TK_CONSTANT sort_expr TK_RPARENgrammar_term : TK_LPAREN TK_VARIABLE sort_expr TK_RPARENgrammar_term : let_grammar_termlet_grammar_term : TK_LPAREN TK_LET grammar_let_binding_plus TK_RPAREN grammar_term TK_RPARENgrammar_let_binding_plus : grammar_let_binding_plus grammar_let_binding\n                                    | grammar_let_bindinggrammar_let_binding : TK_LPAREN symbol sort_expr grammar_term TK_RPARENgrammar_term_star : grammar_term_star grammar_term\n                             | '

_lr_action_items = {'TK_LPAREN': (
[0, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 26, 29, 33, 34, 35, 36, 37, 38, 41, 42, 43, 44, 45, 46, 47,
 48, 49, 50, 52, 53, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 72, 73, 74, 75, 76, 78, 79, 80, 83, 84, 85, 86,
 87, 88, 89, 90, 91, 92, 93, 94, 97, 98, 99, 102, 103, 104, 106, 107, 109, 110, 111, 112, 114, 115, 116, 117, 118, 119,
 123, 124, 129, 131, 132, 134, 135, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 149, 151, 153, 155, 156, 157, 161,
 162, 163, 164, 166, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 179, 180, 184, 185, 186, 187, 188, 189, 190, 191,
 192, 193, 194, 197, 198, ],
[4, 18, 18, -17, -1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, 18, -16, 39, 53, 58, -51, 60, 58, 58, -20, -55, -56, -60,
 -28, -29, -31, -33, -34, -36, -67, 70, 77, 70, 70, -26, -40, 70, -70, 70, 87, -66, 90, 90, 93, -21, -37, -41, -42, -46,
 -50, 77, -73, -30, 105, 39, 70, 111, 113, -24, 39, 105, 39, 39, 122, -68, 127, 70, -27, -72, -22, -32, -38, -39, 70,
 -69, 113, -19, 113, -76, -57, -65, 105, -45, 122, -63, 70, 70, -14, 113, 70, -75, -43, -44, -58, -59, 70, 39, -62, -25,
 -35, -47, -74, -15, 164, -23, 39, -49, -48, -18, 167, -61, 167, -79, -80, -81, -85, -64, -91, 70, 70, 183, -78, 167,
 183, -88, -77, -82, -90, -83, -84, 70, 167, -87, 167, -86, -89, ]), '$end': (
[1, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 38, 67, 80, 88, 99, 103, 104, 112, 132, 145, 153, 156, 163, ],
[0, -13, -17, -1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12, -16, -20, -21, -30, -24, -27, -22, -32, -19, -14, -25,
 -15, -23, -18, ]), 'TK_SET_LOGIC': ([4, ], [20, ]), 'TK_DEFINE_FUN': ([4, 18, ], [21, 21, ]),
                    'TK_DECLARE_FUN': ([4, 18, ], [22, 22, ]), 'TK_SYNTH_FUN': ([4, 18, ], [23, 23, ]),
                    'TK_SYNTH_INV': ([4, 18, ], [24, 24, ]), 'TK_CHECK_SYNTH': ([4, 18, ], [25, 25, ]),
                    'TK_CONSTRAINT': ([4, 18, ], [26, 26, ]), 'TK_INV_CONSTRAINT': ([4, 18, ], [27, 27, ]),
                    'TK_DEFINE_SORT': ([4, 18, ], [28, 28, ]), 'TK_SET_OPTIONS': ([4, 18, ], [29, 29, ]),
                    'TK_DECLARE_VAR': ([4, 18, ], [30, 30, ]), 'TK_DECLARE_PRIMED_VAR': ([4, 18, ], [31, 31, ]),
                    'TK_SYMBOL': (
                    [20, 21, 22, 23, 24, 26, 27, 28, 30, 31, 34, 39, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 55,
                     56, 59, 60, 61, 63, 68, 69, 72, 73, 74, 75, 76, 77, 84, 85, 89, 91, 92, 94, 95, 98, 105, 106, 109,
                     110, 113, 116, 117, 122, 127, 129, 131, 135, 138, 140, 141, 142, 143, 146, 147, 148, 149, 157, 160,
                     161, 164, 166, 167, 168, 169, 170, 171, 172, 174, 175, 176, 179, 180, 183, 187, 188, 189, 190, 191,
                     192, 194, 197, ],
                    [32, 34, 34, 34, 34, 42, 51, 34, 55, 34, -51, 34, -55, -56, -60, -28, -29, -31, -33, -34, -36, -67,
                     69, 76, 76, 76, 76, -70, 76, -66, 94, 95, -37, -41, -42, -46, -50, 34, 42, 76, 42, 42, 42, -68,
                     125, 76, 131, -38, 76, -69, 34, -57, -65, 34, 34, 76, 76, 76, -43, -58, -59, 76, 42, -35, 34, -54,
                     -47, 42, -53, -49, 170, -61, 34, 170, -79, -80, -81, -85, -91, 76, 76, -78, 170, 34, -82, -90, -83,
                     -84, 76, 170, 170, -86, ]), 'TK_RPAREN': (
    [25, 32, 34, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 54, 58, 60, 62, 63, 71, 72, 73, 74, 75, 76, 78, 79, 81, 82,
     83, 85, 86, 89, 94, 101, 102, 106, 107, 108, 110, 114, 115, 116, 117, 118, 119, 120, 121, 123, 124, 125, 126, 128,
     130, 133, 134, 136, 137, 139, 140, 141, 144, 146, 147, 148, 149, 150, 151, 152, 154, 158, 159, 160, 161, 162, 165,
     166, 168, 169, 170, 171, 172, 173, 174, 178, 179, 180, 181, 182, 184, 185, 186, 187, 188, 189, 190, 193, 195, 196,
     197, 198, ],
    [38, 57, -51, 67, -55, -56, -60, -28, -29, -31, -33, -34, -36, -67, 80, -40, -70, 88, -66, 99, -37, -41, -42, -46,
     -50, 101, -73, 103, 104, 106, 109, 112, 116, -68, -71, -72, -38, -39, 132, -69, 136, -76, -57, -65, 138, -45, 140,
     141, 143, -63, 145, 146, 149, 151, 153, 154, 156, -75, -44, -58, -59, -62, -35, 159, -54, -47, 161, -74, 162, 163,
     166, -52, -53, -49, -48, 173, -61, 178, -79, -80, -81, -85, -64, -91, 186, -78, 187, 189, 190, 192, -88, -77, -82,
     -90, -83, -84, -87, 197, 198, -86, -89, ]), 'TK_NUMERAL': (
    [26, 34, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 63, 72, 73, 74, 75, 76, 84, 89, 91, 92, 94, 96, 116, 117, 138, 140,
     141, 143, 146, 149, 157, 161, 164, 166, 168, 169, 170, 171, 172, 174, 179, 180, 187, 188, 189, 190, 192, 194,
     197, ],
    [44, -51, -55, -56, -60, -28, -29, -31, -33, -34, -36, -67, -66, -37, -41, -42, -46, -50, 44, 44, 44, 44, -68, 126,
     -57, -65, -43, -58, -59, 44, -35, -47, 44, -49, 44, -61, 44, -79, -80, -81, -85, -91, -78, 44, -82, -90, -83, -84,
     44, 44, -86, ]), 'TK_BOOL_CONST': (
    [26, 34, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 63, 72, 73, 74, 75, 76, 84, 89, 91, 92, 94, 116, 117, 138, 140,
     141, 143, 146, 149, 157, 161, 164, 166, 168, 169, 170, 171, 172, 174, 179, 180, 187, 188, 189, 190, 192, 194,
     197, ],
    [45, -51, -55, -56, -60, -28, -29, -31, -33, -34, -36, -67, -66, -37, -41, -42, -46, -50, 45, 45, 45, 45, -68, -57,
     -65, -43, -58, -59, 45, -35, -47, 45, -49, 45, -61, 45, -79, -80, -81, -85, -91, -78, 45, -82, -90, -83, -84, 45,
     45, -86, ]), 'TK_DECIMAL': (
    [26, 34, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 63, 72, 73, 74, 75, 76, 84, 89, 91, 92, 94, 116, 117, 138, 140,
     141, 143, 146, 149, 157, 161, 164, 166, 168, 169, 170, 171, 172, 174, 179, 180, 187, 188, 189, 190, 192, 194,
     197, ],
    [46, -51, -55, -56, -60, -28, -29, -31, -33, -34, -36, -67, -66, -37, -41, -42, -46, -50, 46, 46, 46, 46, -68, -57,
     -65, -43, -58, -59, 46, -35, -47, 46, -49, 46, -61, 46, -79, -80, -81, -85, -91, -78, 46, -82, -90, -83, -84, 46,
     46, -86, ]), 'TK_HEX_CONST': (
    [26, 34, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 63, 72, 73, 74, 75, 76, 84, 89, 91, 92, 94, 116, 117, 138, 140,
     141, 143, 146, 149, 157, 161, 164, 166, 168, 169, 170, 171, 172, 174, 179, 180, 187, 188, 189, 190, 192, 194,
     197, ],
    [47, -51, -55, -56, -60, -28, -29, -31, -33, -34, -36, -67, -66, -37, -41, -42, -46, -50, 47, 47, 47, 47, -68, -57,
     -65, -43, -58, -59, 47, -35, -47, 47, -49, 47, -61, 47, -79, -80, -81, -85, -91, -78, 47, -82, -90, -83, -84, 47,
     47, -86, ]), 'TK_BIN_CONST': (
    [26, 34, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 63, 72, 73, 74, 75, 76, 84, 89, 91, 92, 94, 116, 117, 138, 140,
     141, 143, 146, 149, 157, 161, 164, 166, 168, 169, 170, 171, 172, 174, 179, 180, 187, 188, 189, 190, 192, 194,
     197, ],
    [48, -51, -55, -56, -60, -28, -29, -31, -33, -34, -36, -67, -66, -37, -41, -42, -46, -50, 48, 48, 48, 48, -68, -57,
     -65, -43, -58, -59, 48, -35, -47, 48, -49, 48, -61, 48, -79, -80, -81, -85, -91, -78, 48, -82, -90, -83, -84, 48,
     48, -86, ]), 'TK_STRING_LITERAL': (
    [26, 34, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 63, 72, 73, 74, 75, 76, 84, 89, 91, 92, 94, 100, 116, 117, 138,
     140, 141, 143, 146, 149, 157, 161, 164, 166, 168, 169, 170, 171, 172, 174, 179, 180, 187, 188, 189, 190, 192, 194,
     197, ],
    [49, -51, -55, -56, -60, -28, -29, -31, -33, -34, -36, -67, -66, -37, -41, -42, -46, -50, 49, 49, 49, 49, -68, 130,
     -57, -65, -43, -58, -59, 49, -35, -47, 49, -49, 49, -61, 49, -79, -80, -81, -85, -91, -78, 49, -82, -90, -83, -84,
     49, 49, -86, ]), 'TK_INT': (
    [34, 52, 55, 56, 59, 60, 61, 72, 73, 74, 75, 76, 85, 98, 106, 109, 110, 129, 131, 135, 142, 146, 149, 161, 175, 176,
     191, ],
    [-51, 72, 72, 72, 72, -70, 72, -37, -41, -42, -46, -50, 72, 72, -38, 72, -69, 72, 72, 72, 72, -35, -47, -49, 72, 72,
     72, ]), 'TK_BOOL': (
    [34, 52, 55, 56, 59, 60, 61, 72, 73, 74, 75, 76, 85, 98, 106, 109, 110, 129, 131, 135, 142, 146, 149, 161, 175, 176,
     191, ],
    [-51, 73, 73, 73, 73, -70, 73, -37, -41, -42, -46, -50, 73, 73, -38, 73, -69, 73, 73, 73, 73, -35, -47, -49, 73, 73,
     73, ]), 'TK_REAL': (
    [34, 52, 55, 56, 59, 60, 61, 72, 73, 74, 75, 76, 85, 98, 106, 109, 110, 129, 131, 135, 142, 146, 149, 161, 175, 176,
     191, ],
    [-51, 74, 74, 74, 74, -70, 74, -37, -41, -42, -46, -50, 74, 74, -38, 74, -69, 74, 74, 74, 74, -35, -47, -49, 74, 74,
     74, ]), 'TK_STRING': (
    [34, 52, 55, 56, 59, 60, 61, 72, 73, 74, 75, 76, 85, 98, 106, 109, 110, 129, 131, 135, 142, 146, 149, 161, 175, 176,
     191, ],
    [-51, 75, 75, 75, 75, -70, 75, -37, -41, -42, -46, -50, 75, 75, -38, 75, -69, 75, 75, 75, 75, -35, -47, -49, 75, 75,
     75, ]), 'TK_EXISTS': ([39, ], [64, ]), 'TK_FORALL': ([39, ], [65, ]), 'TK_LET': ([39, 167, ], [66, 177, ]),
                    'TK_DOUBLE_COLON': ([42, 170, ], [68, 68, ]), 'TK_BV': ([70, ], [96, ]),
                    'TK_ENUM': ([70, ], [97, ]), 'TK_ARRAY': ([70, ], [98, ]), 'TK_CONSTANT': ([167, ], [175, ]),
                    'TK_VARIABLE': ([167, ], [176, ]), }

_lr_action = {}
for _k, _v in _lr_action_items.items():
    for _x, _y in zip(_v[0], _v[1]):
        if not _x in _lr_action:  _lr_action[_x] = {}
        _lr_action[_x][_k] = _y
del _lr_action_items

_lr_goto_items = {'program': ([0, ], [1, ]), 'set_logic_command': ([0, ], [2, ]), 'command_plus': ([0, 2, ], [3, 17, ]),
                  'command': ([0, 2, 3, 17, ], [5, 5, 19, 19, ]), 'fun_def_command': ([0, 2, 3, 17, ], [6, 6, 6, 6, ]),
                  'fun_decl_command': ([0, 2, 3, 17, ], [7, 7, 7, 7, ]),
                  'synth_fun_command': ([0, 2, 3, 17, ], [8, 8, 8, 8, ]),
                  'synth_inv_command': ([0, 2, 3, 17, ], [9, 9, 9, 9, ]),
                  'check_synth_command': ([0, 2, 3, 17, ], [10, 10, 10, 10, ]),
                  'constraint_command': ([0, 2, 3, 17, ], [11, 11, 11, 11, ]),
                  'inv_constraint_command': ([0, 2, 3, 17, ], [12, 12, 12, 12, ]),
                  'sort_def_command': ([0, 2, 3, 17, ], [13, 13, 13, 13, ]),
                  'set_opts_command': ([0, 2, 3, 17, ], [14, 14, 14, 14, ]),
                  'var_decl_command': ([0, 2, 3, 17, ], [15, 15, 15, 15, ]),
                  'primed_var_decl_command': ([0, 2, 3, 17, ], [16, 16, 16, 16, ]), 'symbol': (
    [21, 22, 23, 24, 28, 31, 39, 77, 113, 122, 127, 147, 167, 183, ],
    [33, 35, 36, 37, 52, 56, 63, 100, 135, 142, 148, 160, 174, 191, ]),
                  'term': ([26, 84, 89, 91, 92, 143, 157, ], [40, 108, 117, 120, 121, 158, 165, ]), 'literal': (
    [26, 84, 89, 91, 92, 143, 157, 164, 168, 180, 192, 194, ], [41, 41, 41, 41, 41, 41, 41, 171, 171, 171, 171, 171, ]),
                  'let_term': ([26, 84, 89, 91, 92, 143, 157, ], [43, 43, 43, 43, 43, 43, 43, ]), 'enum_const': (
    [26, 84, 89, 91, 92, 143, 157, 164, 168, 180, 192, 194, ], [50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, ]),
                  'option_list': ([29, ], [54, ]), 'arg_list': ([33, 36, 37, ], [59, 61, 62, ]), 'sort_expr': (
    [52, 55, 56, 59, 61, 85, 98, 109, 129, 131, 135, 142, 175, 176, 191, ],
    [71, 81, 82, 84, 86, 110, 129, 133, 150, 152, 155, 157, 181, 182, 194, ]), 'symbol_pair_plus': ([53, ], [78, ]),
                  'symbol_pair': ([53, 78, ], [79, 102, ]), 'sorted_symbol_star': ([58, ], [83, ]),
                  'sort_star': ([60, ], [85, ]), 'term_star': ([63, ], [89, ]),
                  'nonempty_arg_list': ([64, 65, ], [91, 92, ]), 'sorted_symbol': ([83, 90, 118, ], [107, 119, 139, ]),
                  'nonterminal_def_plus': ([87, 111, ], [114, 134, ]),
                  'nonterminal_def': ([87, 111, 114, 134, ], [115, 115, 137, 137, ]),
                  'sorted_symbol_plus': ([90, ], [118, ]), 'binding_plus': ([93, ], [123, ]),
                  'binding': ([93, 123, ], [124, 144, ]), 'enum_constructor_list': ([97, ], [128, ]),
                  'symbol_plus': ([127, ], [147, ]), 'grammar_term_plus': ([164, ], [168, ]),
                  'grammar_term': ([164, 168, 180, 192, 194, ], [169, 179, 188, 195, 196, ]),
                  'let_grammar_term': ([164, 168, 180, 192, 194, ], [172, 172, 172, 172, 172, ]),
                  'grammar_term_star': ([174, ], [180, ]), 'grammar_let_binding_plus': ([177, ], [184, ]),
                  'grammar_let_binding': ([177, 184, ], [185, 193, ]), }

_lr_goto = {}
for _k, _v in _lr_goto_items.items():
    for _x, _y in zip(_v[0], _v[1]):
        if not _x in _lr_goto: _lr_goto[_x] = {}
        _lr_goto[_x][_k] = _y
del _lr_goto_items
_lr_productions = [
    ("S' -> program", "S'", 1, None, None, None),
    ('command -> fun_def_command', 'command', 1, 'p_command', 'parser.py', 11),
    ('command -> fun_decl_command', 'command', 1, 'p_command', 'parser.py', 12),
    ('command -> synth_fun_command', 'command', 1, 'p_command', 'parser.py', 13),
    ('command -> synth_inv_command', 'command', 1, 'p_command', 'parser.py', 14),
    ('command -> check_synth_command', 'command', 1, 'p_command', 'parser.py', 15),
    ('command -> constraint_command', 'command', 1, 'p_command', 'parser.py', 16),
    ('command -> inv_constraint_command', 'command', 1, 'p_command', 'parser.py', 17),
    ('command -> sort_def_command', 'command', 1, 'p_command', 'parser.py', 18),
    ('command -> set_opts_command', 'command', 1, 'p_command', 'parser.py', 19),
    ('command -> var_decl_command', 'command', 1, 'p_command', 'parser.py', 20),
    ('command -> primed_var_decl_command', 'command', 1, 'p_command', 'parser.py', 21),
    ('program -> set_logic_command command_plus', 'program', 2, 'p_program', 'parser.py', 20),
    ('program -> command_plus', 'program', 1, 'p_program', 'parser.py', 21),
    ('fun_def_command -> TK_LPAREN TK_DEFINE_FUN symbol arg_list sort_expr term TK_RPAREN', 'fun_def_command', 7,
     'p_fun_def_command', 'parser.py', 25),
    ('fun_decl_command -> TK_LPAREN TK_DECLARE_FUN symbol TK_LPAREN sort_star TK_RPAREN sort_expr TK_RPAREN',
     'fun_decl_command', 8, 'p_fun_decl_command', 'parser.py', 31),
    ('command_plus -> command_plus command', 'command_plus', 2, 'p_command_plus', 'parser.py', 32),
    ('command_plus -> command', 'command_plus', 1, 'p_command_plus', 'parser.py', 33),
    (
    'synth_fun_command -> TK_LPAREN TK_SYNTH_FUN symbol arg_list sort_expr TK_LPAREN nonterminal_def_plus TK_RPAREN TK_RPAREN',
    'synth_fun_command', 9, 'p_synth_fun_command', 'parser.py', 37),
    ('synth_fun_command -> TK_LPAREN TK_SYNTH_FUN symbol arg_list sort_expr TK_RPAREN', 'synth_fun_command', 6,
     'p_synth_fun_command', 'parser.py', 38),
    ('check_synth_command -> TK_LPAREN TK_CHECK_SYNTH TK_RPAREN', 'check_synth_command', 3, 'p_check_synth_command',
     'parser.py', 40),
    ('constraint_command -> TK_LPAREN TK_CONSTRAINT term TK_RPAREN', 'constraint_command', 4, 'p_constraint_command',
     'parser.py', 46),
    ('var_decl_command -> TK_LPAREN TK_DECLARE_VAR TK_SYMBOL sort_expr TK_RPAREN', 'var_decl_command', 5,
     'p_var_decl_command', 'parser.py', 52),
    ('synth_inv_command -> TK_LPAREN TK_SYNTH_INV symbol arg_list TK_LPAREN nonterminal_def_plus TK_RPAREN TK_RPAREN',
     'synth_inv_command', 8, 'p_synth_inv_command', 'parser.py', 55),
    ('synth_inv_command -> TK_LPAREN TK_SYNTH_INV symbol arg_list TK_RPAREN', 'synth_inv_command', 5,
     'p_synth_inv_command', 'parser.py', 56),
    ('inv_constraint_command -> TK_LPAREN TK_INV_CONSTRAINT TK_SYMBOL TK_SYMBOL TK_SYMBOL TK_SYMBOL TK_RPAREN',
     'inv_constraint_command', 7, 'p_inv_constraint_command', 'parser.py', 58),
    ('set_logic_command -> TK_LPAREN TK_SET_LOGIC TK_SYMBOL TK_RPAREN', 'set_logic_command', 4, 'p_set_logic_command',
     'parser.py', 64),
    ('sort_def_command -> TK_LPAREN TK_DEFINE_SORT symbol sort_expr TK_RPAREN', 'sort_def_command', 5,
     'p_sort_def_command', 'parser.py', 73),
    ('literal -> TK_NUMERAL', 'literal', 1, 'p_literal_numeral', 'parser.py', 75),
    ('literal -> TK_BOOL_CONST', 'literal', 1, 'p_literal_bool_const', 'parser.py', 79),
    ('set_opts_command -> TK_LPAREN TK_SET_OPTIONS option_list TK_RPAREN', 'set_opts_command', 4, 'p_set_opts_command',
     'parser.py', 79),
    ('literal -> TK_DECIMAL', 'literal', 1, 'p_literal_decimal', 'parser.py', 83),
    ('primed_var_decl_command -> TK_LPAREN TK_DECLARE_PRIMED_VAR symbol sort_expr TK_RPAREN', 'primed_var_decl_command',
     5, 'p_primed_var_decl_command', 'parser.py', 85),
    ('literal -> TK_HEX_CONST', 'literal', 1, 'p_literal_hex_const', 'parser.py', 87),
    ('literal -> TK_BIN_CONST', 'literal', 1, 'p_literal_bin_const', 'parser.py', 91),
    ('sort_expr -> TK_LPAREN TK_BV TK_NUMERAL TK_RPAREN', 'sort_expr', 4, 'p_sort_expr_bit_vector', 'parser.py', 91),
    ('literal -> TK_STRING_LITERAL', 'literal', 1, 'p_literal_string', 'parser.py', 95),
    ('sort_expr -> TK_INT', 'sort_expr', 1, 'p_sort_expr_int', 'parser.py', 97),
    ('arg_list -> TK_LPAREN sorted_symbol_star TK_RPAREN', 'arg_list', 3, 'p_arg_list', 'parser.py', 99),
    ('sorted_symbol_star -> sorted_symbol_star sorted_symbol', 'sorted_symbol_star', 2, 'p_sorted_symbol_star',
     'parser.py', 103),
    ('sorted_symbol_star -> <empty>', 'sorted_symbol_star', 0, 'p_sorted_symbol_star', 'parser.py', 104),
    ('sort_expr -> TK_BOOL', 'sort_expr', 1, 'p_sort_expr_bool', 'parser.py', 103),
    ('sort_expr -> TK_REAL', 'sort_expr', 1, 'p_sort_expr_real', 'parser.py', 109),
    ('nonempty_arg_list -> TK_LPAREN sorted_symbol_plus TK_RPAREN', 'nonempty_arg_list', 3, 'p_nonempty_arg_list',
     'parser.py', 111),
    ('sorted_symbol_plus -> sorted_symbol_plus sorted_symbol', 'sorted_symbol_plus', 2, 'p_sorted_symbol_plus',
     'parser.py', 115),
    ('sorted_symbol_plus -> sorted_symbol', 'sorted_symbol_plus', 1, 'p_sorted_symbol_plus', 'parser.py', 116),
    ('sort_expr -> TK_STRING', 'sort_expr', 1, 'p_sort_expr_string', 'parser.py', 115),
    ('sort_expr -> TK_LPAREN TK_ENUM enum_constructor_list TK_RPAREN', 'sort_expr', 4, 'p_sort_expr_enum', 'parser.py',
     121),
    ('sorted_symbol -> TK_LPAREN TK_SYMBOL sort_expr TK_RPAREN', 'sorted_symbol', 4, 'p_sorted_symbol', 'parser.py',
     123),
    ('sort_expr -> TK_LPAREN TK_ARRAY sort_expr sort_expr TK_RPAREN', 'sort_expr', 5, 'p_sort_expr_array', 'parser.py',
     127),
    ('sort_expr -> TK_SYMBOL', 'sort_expr', 1, 'p_sort_expr_symbol', 'parser.py', 133),
    ('symbol -> TK_SYMBOL', 'symbol', 1, 'p_symbol', 'parser.py', 139),
    ('enum_constructor_list -> TK_LPAREN symbol_plus TK_RPAREN', 'enum_constructor_list', 3, 'p_enum_constructor_list',
     'parser.py', 143),
    ('symbol_plus -> symbol_plus symbol', 'symbol_plus', 2, 'p_symbol_plus_close_paren', 'parser.py', 147),
    ('symbol_plus -> symbol', 'symbol_plus', 1, 'p_symbol_plus_close_paren', 'parser.py', 148),
    ('term -> literal', 'term', 1, 'p_term', 'parser.py', 155),
    ('term -> TK_SYMBOL', 'term', 1, 'p_term', 'parser.py', 156),
    ('term -> TK_LPAREN symbol term_star TK_RPAREN', 'term', 4, 'p_term', 'parser.py', 157),
    ('term -> TK_LPAREN TK_EXISTS nonempty_arg_list term TK_RPAREN', 'term', 5, 'p_term', 'parser.py', 158),
    ('term -> TK_LPAREN TK_FORALL nonempty_arg_list term TK_RPAREN', 'term', 5, 'p_term', 'parser.py', 159),
    ('term -> let_term', 'term', 1, 'p_term', 'parser.py', 160),
    ('let_term -> TK_LPAREN TK_LET TK_LPAREN binding_plus TK_RPAREN term TK_RPAREN', 'let_term', 7, 'p_let_term',
     'parser.py', 180),
    ('binding_plus -> binding_plus binding', 'binding_plus', 2, 'p_binding_plus', 'parser.py', 188),
    ('binding_plus -> binding', 'binding_plus', 1, 'p_binding_plus', 'parser.py', 189),
    ('binding -> TK_LPAREN symbol sort_expr term TK_RPAREN', 'binding', 5, 'p_binding', 'parser.py', 196),
    ('term_star -> term_star term', 'term_star', 2, 'p_term_star', 'parser.py', 200),
    ('term_star -> <empty>', 'term_star', 0, 'p_term_star', 'parser.py', 201),
    ('literal -> enum_const', 'literal', 1, 'p_literal_enum_const', 'parser.py', 208),
    ('enum_const -> TK_SYMBOL TK_DOUBLE_COLON TK_SYMBOL', 'enum_const', 3, 'p_enum_const', 'parser.py', 212),
    ('sort_star -> sort_star sort_expr', 'sort_star', 2, 'p_sort_star', 'parser.py', 216),
    ('sort_star -> <empty>', 'sort_star', 0, 'p_sort_star', 'parser.py', 217),
    ('option_list -> TK_LPAREN symbol_pair_plus TK_RPAREN', 'option_list', 3, 'p_option_list', 'parser.py', 224),
    ('symbol_pair_plus -> symbol_pair_plus symbol_pair', 'symbol_pair_plus', 2, 'p_symbol_pair_plus', 'parser.py', 228),
    ('symbol_pair_plus -> symbol_pair', 'symbol_pair_plus', 1, 'p_symbol_pair_plus', 'parser.py', 229),
    (
    'symbol_pair -> TK_LPAREN symbol TK_STRING_LITERAL TK_RPAREN', 'symbol_pair', 4, 'p_symbol_pair', 'parser.py', 236),
    (
    'nonterminal_def_plus -> nonterminal_def_plus nonterminal_def', 'nonterminal_def_plus', 2, 'p_nonterminal_def_plus',
    'parser.py', 240),
    ('nonterminal_def_plus -> nonterminal_def', 'nonterminal_def_plus', 1, 'p_nonterminal_def_plus', 'parser.py', 241),
    ('nonterminal_def -> TK_LPAREN symbol sort_expr TK_LPAREN grammar_term_plus TK_RPAREN TK_RPAREN', 'nonterminal_def',
     7, 'p_nonterminal_def', 'parser.py', 248),
    ('grammar_term_plus -> grammar_term_plus grammar_term', 'grammar_term_plus', 2, 'p_grammar_term_plus', 'parser.py',
     254),
    ('grammar_term_plus -> grammar_term', 'grammar_term_plus', 1, 'p_grammar_term_plus', 'parser.py', 255),
    ('grammar_term -> TK_SYMBOL', 'grammar_term', 1, 'p_grammar_term_symbol', 'parser.py', 262),
    ('grammar_term -> literal', 'grammar_term', 1, 'p_grammar_term_literal', 'parser.py', 269),
    ('grammar_term -> TK_LPAREN symbol grammar_term_star TK_RPAREN', 'grammar_term', 4,
     'p_grammar_term_function_application', 'parser.py', 274),
    ('grammar_term -> TK_LPAREN TK_CONSTANT sort_expr TK_RPAREN', 'grammar_term', 4, 'p_grammar_term_constant',
     'parser.py', 281),
    ('grammar_term -> TK_LPAREN TK_VARIABLE sort_expr TK_RPAREN', 'grammar_term', 4, 'p_grammar_term_variable',
     'parser.py', 287),
    ('grammar_term -> let_grammar_term', 'grammar_term', 1, 'p_grammar_term_let', 'parser.py', 293),
    ('let_grammar_term -> TK_LPAREN TK_LET grammar_let_binding_plus TK_RPAREN grammar_term TK_RPAREN',
     'let_grammar_term', 6, 'p_let_grammar_term', 'parser.py', 297),
    ('grammar_let_binding_plus -> grammar_let_binding_plus grammar_let_binding', 'grammar_let_binding_plus', 2,
     'p_grammar_let_binding_plus', 'parser.py', 303),
    ('grammar_let_binding_plus -> grammar_let_binding', 'grammar_let_binding_plus', 1, 'p_grammar_let_binding_plus',
     'parser.py', 304),
    ('grammar_let_binding -> TK_LPAREN symbol sort_expr grammar_term TK_RPAREN', 'grammar_let_binding', 5,
     'p_grammar_let_binding', 'parser.py', 311),
    ('grammar_term_star -> grammar_term_star grammar_term', 'grammar_term_star', 2, 'p_grammar_term_star', 'parser.py',
     315),
    ('grammar_term_star -> <empty>', 'grammar_term_star', 0, 'p_grammar_term_star', 'parser.py', 316),
]
