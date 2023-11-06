# DFS

Notes

#brute force
 travel down tree, if not true, go back up
if true, return the theory
 if ever reach root node, prob is unsat and return

recursively propagate boolean constraints - don't have to traverse the whole tree before backtracking

DPLL

VSIDS
2 watched literals

CDCL - conflict driven clause learning
- implication graph - like a dependency graph
-isolate conflict with a cut - can be moved to different set of literals
- cut separates reason side and conflict side
- use contrapositive to negate conflict side
- demorgan laws - if negate an AND & an OR clause, flip the operator and negate all the other literals in the expression
- clause maintenance, minimisation, resolution, phase saving


SMT approach - establish structure of phi/problem via component atoms
establish incompatible relation between atoms

Eager - slow
- encoder - takes in SMT formula and returns a sat-solver formula including incompatibilities to the sat solver
- sat solver - determines if sat/unsat

Lazy
- structure extractor - takes smt formula and passes onto a SAT solver in CNF form
- sat solver e.g. DPLL - if sat -> passes the candidate model to a T-solver, if compatible, returns sat else returns negative signal to SAT-solver
- the sat solver will then add the negation of the model into the new formula as an additional constraint

Unit propogation - setting one unit to true will remove that unit clause from the formula,
then chose a literal and set it to a value until all clauses or literals have values
then we take this model and check if it is consistent in the T-solver
if this is not consistent, add the negation of the model to the original formula
now as the old model doesn't satisfy the new formula, backtrack until we can develop a new model


"""