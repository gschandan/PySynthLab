from _ast import Constant, Name, AST
import z3
import random
from typing import List, Tuple, Union, Any, Optional
import ast


class TopDownCandidateGenerator:
    def __init__(self, problem: 'SynthesisProblem'):
        self.problem = problem
        self.min_const = problem.options.synthesis_parameters.min_const
        self.max_const = problem.options.synthesis_parameters.max_const
        self.max_depth = problem.options.synthesis_parameters.max_depth
        self.grammar = self.define_grammar()
        self.explored_expressions: set[str] = set()

    def define_grammar(self):
        return {
            'EXPR': ['ARITH', 'IF'],
            'ARITH': ['ADD', 'SUB', 'MUL', 'VAR', 'CONST'],
            'ADD': [ast.BinOp(left=ast.Name(id='ARITH', ctx=ast.Load()), op=ast.Add(), right=ast.Name(id='ARITH', ctx=ast.Load()))],
            'SUB': [ast.BinOp(left=ast.Name(id='ARITH', ctx=ast.Load()), op=ast.Sub(), right=ast.Name(id='ARITH', ctx=ast.Load()))],
            'MUL': [ast.BinOp(left=ast.Name(id='ARITH', ctx=ast.Load()), op=ast.Mult(), right=ast.Name(id='CONST', ctx=ast.Load()))],
            'IF': [ast.IfExp(test=ast.Name(id='COND', ctx=ast.Load()), body=ast.Name(id='EXPR', ctx=ast.Load()), orelse=ast.Name(id='EXPR', ctx=ast.Load()))],
            'COND': [
                ast.Compare(left=ast.Name(id='ARITH', ctx=ast.Load()), ops=[ast.Lt()], comparators=[ast.Name(id='ARITH', ctx=ast.Load())]),
                ast.Compare(left=ast.Name(id='ARITH', ctx=ast.Load()), ops=[ast.LtE()], comparators=[ast.Name(id='ARITH', ctx=ast.Load())]),
                ast.Compare(left=ast.Name(id='ARITH', ctx=ast.Load()), ops=[ast.Gt()], comparators=[ast.Name(id='ARITH', ctx=ast.Load())]),
                ast.Compare(left=ast.Name(id='ARITH', ctx=ast.Load()), ops=[ast.GtE()], comparators=[ast.Name(id='ARITH', ctx=ast.Load())]),
                ast.Compare(left=ast.Name(id='ARITH', ctx=ast.Load()), ops=[ast.Eq()], comparators=[ast.Name(id='ARITH', ctx=ast.Load())])
            ],
            'VAR': [ast.Name(id=var) for var in self.problem.context.variable_mapping_dict.keys()],
            'CONST': [ast.Constant(value=const) for const in range(self.min_const, self.max_const + 1)]
        }

    def generate_candidates(self) -> List[Tuple[z3.FuncDeclRef, z3.ExprRef]]:
        candidates = []
        for func_name in self.problem.context.variable_mapping_dict.keys():
            attempts = 0
            max_attempts = 20
            while attempts < max_attempts:
                candidate = self.generate_term('EXPR', func_name)
                if candidate is not None and str(candidate) not in self.explored_expressions:
                    z3_expr = self.ast_to_z3(candidate, func_name)
                    simplified_candidate = self.simplify_term(z3_expr)
                    self.problem.logger.debug(f"Simplified candidate for {func_name}: {simplified_candidate}")
                    func_decl = self.problem.context.variable_mapping_dict[func_name]
                    candidates.append((func_decl, simplified_candidate))
                    self.explored_expressions.add(str(candidate))
                    break
                attempts += 1
    
            if attempts == max_attempts:
                self.problem.logger.warning(
                    f"Couldn't generate valid candidate for {func_name} after {max_attempts} attempts"
                )
        self.problem.logger.debug(f"Generated {len(candidates)} candidates: {candidates}")
        return candidates

    def generate_term(self, symbol: str, func_name: str, depth: int = 0) -> Optional[ast.AST]:
        if depth > self.max_depth:
            return None

        productions = self.grammar[symbol]
        random.shuffle(productions) 

        for production in productions:
            expanded = self.expand_production(production, func_name, depth)
            if expanded is not None:
                return expanded

        self.problem.logger.debug(f"Failed to generate term for symbol: {symbol} at depth: {depth}")
        return None

    def expand_production(self, production: Union[ast.AST, str], func_name: str, depth: int) -> Optional[ast.AST]:
        self.problem.logger.debug(f"Expanding production: {ast.dump(production) if isinstance(production, ast.AST) else production}")

        if isinstance(production, str):
            return self.generate_term(production, func_name, depth + 1)
    
        if isinstance(production, ast.Constant):
            return production
    
        if isinstance(production, ast.Name):
            if production.id in self.problem.context.variable_mapping_dict[func_name]:
                return production
            return self.generate_term(production.id, func_name, depth + 1)
    
        if isinstance(production, (ast.BinOp, ast.Compare, ast.IfExp)):
            new_node = ast.copy_location(production, production)
            for field, value in ast.iter_fields(new_node):
                if isinstance(value, ast.AST):
                    new_value = self.expand_production(value, func_name, depth + 1)
                    if new_value is None:
                        self.problem.logger.debug(f"Failed to expand {field} in {ast.dump(production)}")
                        return None
                    setattr(new_node, field, new_value)
                elif isinstance(value, list):
                    new_list = []
                    for item in value:
                        if isinstance(item, ast.AST):
                            new_item = self.expand_production(item, func_name, depth + 1)
                            if new_item is None:
                                self.problem.logger.debug(f"Failed to expand list item in {ast.dump(production)}")
                                return None
                            new_list.append(new_item)
                        else:
                            new_list.append(item)
                    setattr(new_node, field, new_list)
            return new_node
    
        if isinstance(production, (ast.Add, ast.Sub, ast.Mult)):
            return production 
    
        self.problem.logger.warning(f"Unsupported production type: {type(production)}")
        return None

    def ast_to_z3(self, node: ast.AST, func_name: str) -> z3.ExprRef:
        if isinstance(node, ast.BinOp):
            left = self.ast_to_z3(node.left, func_name)
            right = self.ast_to_z3(node.right, func_name)
            if isinstance(node.op, ast.Add):
                return left + right
            elif isinstance(node.op, ast.Sub):
                return left - right
            elif isinstance(node.op, ast.Mult):
                return left * right
        elif isinstance(node, ast.Compare):
            left = self.ast_to_z3(node.left, func_name)
            right = self.ast_to_z3(node.comparators[0], func_name)
            if isinstance(node.ops[0], ast.Lt):
                return left < right
            elif isinstance(node.ops[0], ast.LtE):
                return left <= right
            elif isinstance(node.ops[0], ast.Gt):
                return left > right
            elif isinstance(node.ops[0], ast.GtE):
                return left >= right
            elif isinstance(node.ops[0], ast.Eq):
                return left == right
        elif isinstance(node, ast.IfExp):
            test = self.ast_to_z3(node.test, func_name)
            body = self.ast_to_z3(node.body, func_name)
            orelse = self.ast_to_z3(node.orelse, func_name)
            return z3.If(test, body, orelse)
        elif isinstance(node, ast.Name):
            if node.id == func_name:
                # Return the function declaration instead of a variable
                return self.problem.context.variable_mapping_dict[func_name]
            return z3.Int(node.id)
        elif isinstance(node, ast.Constant):
            return z3.IntVal(node.value)
        raise ValueError(f"Unsupported AST node: {ast.dump(node)}")

    def simplify_term(self, term: Union[z3.ExprRef, int]) -> Union[z3.ExprRef, int]:
        if isinstance(term, z3.ExprRef):
            return z3.simplify(term)
        return term

    def prune_candidates(self, candidates: List[Tuple[z3.ExprRef, str]]) -> List[Tuple[z3.ExprRef, str]]:
        return candidates
