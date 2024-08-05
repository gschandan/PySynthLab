# PySynthLab

PySynthLab is a synthesis engine for solving Linear Integer Arithmetic (LIA) synthesis problems using Counterexample-Guided Inductive Synthesis (CEGIS). 
It provides various strategies and configurations for performing program synthesis.

## Features

- Multiple synthesis strategies: Fast Enumerative, Random Search (Bottom-Up and Top-Down)
- Configurable candidate generation
- Customizable synthesis parameters
- Logging and configuration management

## Prerequisites

- Python 3.8 or higher
- Virtualenv
- z3 Theorem Prover

## Installation

1. Clone the repository:

```shell
git clone https://github.com/gschandan/PySynthLab.git
cd PySynthLab
```

2. Create and activate the virtual environment:

```shell
python -m venv venv
source venv/bin/activate
```

3. Install the requirements without using the venv:
```shell
pip install -r requirements.txt
```
or using Make:
```shell
make init
```
or install using the venv
```shell
./venv/bin/pip install -r requirements.txt  
```
or using the venv via Make
```shell
make init_venv  
```

## Running the Synthesiser
You can run the Synthesiser by providing a path to the SyGuS input file or by providing the input problem via stdin.

Either provide a path to the sygus input file:
```shell
python -m src.runner problems/debugging/small.sl  
```
or provide the input problem via STDIN:
```shell
python -m src.runner - <<EOF
(set-logic LIA)
(synth-fun max2 ((a Int) (b Int)) Int)
(declare-var x Int)
(declare-var y Int)
(constraint (= (f x y) (f y x)))
(constraint (and (<= x (f x y)) (<= y (f x y))))
(check-synth)
EOF
```
or if you have used the venv
```shell
./venv/bin/python -m src.runner problems/debugging/small.sl  
```
```shell
./venv/bin/python -m src.runner <<EOF
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
```
## Configuration
PySynthLab supports configuration through command-line arguments, YAML files, and default options. The configuration is managed by the ConfigManager class.

### Command Line Options

- `--strategy`: Choose the synthesis strategy (fast_enumerative, random_search_bottom_up, random_search_top_down)
- `--candidate-generation`: Candidate generation strategy (bottom_up, top_down, fast_enumerative)
- `--max-iterations`: Maximum number of iterations
- `--max-depth`: Maximum depth of generated expressions
- `--max-complexity`: Maximum complexity of generated expressions
- `--random-seed`: Random seed for solvers
- `--randomise-each`-iteration: Randomize seed between each synthesis iteration
- `--max-candidates`-at-each-depth: Maximum number of candidates to consider at each depth
- `--min-const`, --max-const: Range of constants to introduce into candidate programs
- `--use-weighted`-generator: Use weighted top-down generator
- `--logging-level`: Set logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- `--logging-file`: Specify log file path

For a complete list or further information, run:
```shell
python -m src.runner --help
```
### YAML configuration
You can also use a YAML file for configuration. 
Please see and edit the `user_config.yaml` in `src/config`:
```yaml
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
```

## Docs
Please click [here](https://gschandan.github.io/PySynthLab/) to see the docs

## Run Tests
Run general tests:
```shell
make test
```
Run general tests:
```shell
python -m unittest discover -s tests -p "test_*.py"
```
Run some specific debugging tests built around specific problems:
```shell
python -m tests.run_problem_tests
```

## Run Benchmarks
Edit the file `src/benchmark/run_all_benchmarks.py` with the desired configuration(s) and point it at the relevant folder
of problems, then run:
```shell
./venv/bin/python src/benchmark/run_all_benchmarks.py 
```