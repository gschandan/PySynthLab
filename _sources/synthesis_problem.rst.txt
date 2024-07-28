SynthesisProblem
================

.. automodule:: src.cegis.z3.synthesis_problem
   :undoc-members:
   :show-inheritance:

.. autoclass:: src.cegis.z3.synthesis_problem.SynthesisProblem
   :members:
   :undoc-members:
   :show-inheritance:

   .. automethod:: __init__
   .. automethod:: __str__

.. autoclass:: src.cegis.z3.synthesis_problem.SynthesisProblemContext
   :members:
   :undoc-members:
   :show-inheritance:

Examples
--------

Creating a SynthesisProblem instance:

.. code-block:: python

   problem_str = "(set-logic LIA)\n(synth-fun max2 ((x Int) (y Int)) Int)\n(declare-var a Int)\n(declare-var b Int)\n(constraint (>= (max2 a b) a))"
   synthesis_problem = SynthesisProblem(problem_str)

Getting the logic of the problem:

.. code-block:: python

   logic = synthesis_problem.get_logic()
   print(logic)  # Output: LIA

Getting synthesis functions:

.. code-block:: python

   synth_funcs = synthesis_problem.get_synth_funcs()
   print(list(synth_funcs.keys()))  # Output: ['max2']

Parsing constraints:

.. code-block:: python

   synthesis_problem.parse_constraints()
   print(synthesis_problem.context.z3_constraints)  # Output: [And(max2(a, b) >= a)]

Substituting candidate expressions:

.. code-block:: python

   from z3 import Int, Function
   max2 = synthesis_problem.context.z3_synth_functions['max2']
   a, b = Int('a'), Int('b')
   replacement = lambda x, y: If(x > y, x, y)
   substituted = synthesis_problem.substitute_constraints(synthesis_problem.context.z3_constraints, [max2], [replacement])
   print(substituted[0])  # Output: If(a > b, a, b) >= a

For more detailed examples and usage, refer to the individual method docstrings.