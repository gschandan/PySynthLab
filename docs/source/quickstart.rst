
Quickstart Guide
================

This guide will help you get started with PySynthLab quickly.

Running the Synthesiser
-----------------------

You can run the Synthesiser by providing a path to the SyGuS input file or by providing the input problem via stdin.

1. Using a SyGuS input file:

   .. code-block:: shell

      python -m src.runner problems/debugging/small.sl

   If you're using a virtual environment:

   .. code-block:: shell

      ./venv/bin/python -m src.runner problems/debugging/small.sl

2. Providing input via STDIN:

   .. code-block:: shell

      python -m src.runner - <<EOF
      (set-logic LIA)
      (synth-fun max2 ((a Int) (b Int)) Int)
      (declare-var x Int)
      (declare-var y Int)
      (constraint (>= (max2 x y) x))
      (constraint (>= (max2 x y) y))
      (constraint (or (= x (max2 x y)) (= y (max2 x y))))
      (constraint (= (max2 x x) x))
      (declare-var a Int)
      (declare-var b Int)
      (constraint (=> (and (>= a b) (>= b a)) (= (max2 a b) a)))
      (constraint (=> (and (>= b a) (>= a b)) (= (max2 b a) b)))
      (check-synth)
      EOF

   If you're using a virtual environment:

   .. code-block:: shell

      ./venv/bin/python -m src.runner - <<EOF
      # (same input as above)
      EOF

Configuration
-------------

PySynthLab supports configuration through command-line arguments, YAML files, and default options.

Command Line Options
^^^^^^^^^^^^^^^^^^^^

Here are some key command-line options:

- ``--strategy``: Choose the synthesis strategy (fast_enumerative, random_search_bottom_up, random_search_top_down)
- ``--candidate-generation``: Candidate generation strategy (bottom_up, top_down, fast_enumerative)
- ``--max-iterations``: Maximum number of iterations
- ``--max-depth``: Maximum depth of generated expressions
- ``--logging-level``: Set logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

For a complete list of options, run:

.. code-block:: shell

   python -m src.runner --help

YAML Configuration
^^^^^^^^^^^^^^^^^^

You can also use a YAML file for configuration. Edit the ``user_config.yaml`` in ``src/config``:

.. code-block:: yaml

   logging:
     level: INFO
     file: logs/synthesis.log

   synthesis_parameters:
     strategy: fast_enumerative
     max_iterations: 1000
     max_depth: 5

   solver:
     name: z3
     timeout: 30000

Running Tests
-------------

To run general tests:

.. code-block:: shell

   make test

Or:

.. code-block:: shell

   python -m unittest discover -s tests -p "test_*.py"

To run specific debugging tests:

.. code-block:: shell

   python -m tests.run_problem_tests

Running Benchmarks
------------------

To run benchmarks:

1. Edit the file ``src/benchmark/run_all_benchmarks.py`` with the desired configuration(s) and point it at the relevant folder of problems.

2. Run the benchmarks:

   .. code-block:: shell

      ./venv/bin/python src/benchmark/run_all_benchmarks.py

For more detailed information, please refer to the full documentation.