SynthesisProblem
================

Base Classes
------------

.. automodule:: src.cegis.synthesis_problem_base
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: src.cegis.synthesis_problem_base.BaseSynthesisProblem
   :members:
   :undoc-members:
   :show-inheritance:

   .. automethod:: __init__
   .. automethod:: __str__

Z3 Extensions
-------------

.. automodule:: src.cegis.z3.synthesis_problem_z3
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: src.cegis.z3.synthesis_problem_z3.SynthesisProblemZ3
   :members:
   :undoc-members:
   :show-inheritance:

   .. automethod:: __init__
   .. automethod:: __str__

.. autoclass:: src.cegis.z3.synthesis_problem_z3.SynthesisProblemZ3Context
   :members:
   :undoc-members:
   :show-inheritance:

Examples
--------

Creating a SynthesisProblem instance:

.. code-block:: python

   from src.cegis.z3.synthesis_problem_z3 import SynthesisProblemZ3
   from src.utilities.options import Options

   problem_str = "(set-logic LIA)\n(synth-fun max2 ((x Int) (y Int)) Int)\n(declare-var a Int)\n(declare-var b Int)\n(constraint (>= (max2 a b) a))"
   options = Options()
   synthesis_problem = SynthesisProblemZ3(problem_str, options)

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

   from z3 import Int, Function, If
   max2 = synthesis_problem.context.z3_synth_functions['max2']
   a, b = Int('a'), Int('b')
   replacement = lambda x, y: If(x > y, x, y)
   substituted = synthesis_problem.substitute_constraints([synthesis_problem.context.z3_constraints], [max2], [replacement])
   print(substituted[0])  # Output: If(a > b, a, b) >= a

For more detailed examples and usage, refer to the individual method docstrings of both base classes and Z3 extensions.