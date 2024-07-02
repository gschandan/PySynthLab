mouse# PySynthLab

This project is a synthesis engine for solving LIA synthesis problems using CEGIS. 

## Prerequisites

- Python 3.8 or higher
- Virtualenv
- z3

## Installation

1. Clone the repository:

```shell
git clone git@github.com:gschandan/PySynthLab.git
cd PySynthLab
```

2. Create a virtual environment:

```shell
python -m venv venv
```

3. Activate the virtual environment:

```shell
source venv/bin/activate
```
4. Install the requirements without using the venv:
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
python src/runner.py -s 1 /path_to_file/small.sl  
```
or provide the input problem via STDIN:
```shell
python -m src.runner -s 1 - <<EOF
(set-logic LIA)
(synth-fun f ((x Int) (y Int)) Int)
(declare-var x Int)
(declare-var y Int)
(constraint (= (f x y) (f y x)))
(constraint (and (<= x (f x y)) (<= y (f x y))))
(check-synth)
EOF
```
or if you have used the venv
```shell
./venv/bin/python src/runner.py -s 1 /path_to_file/small.sl  
```
```shell
./venv/bin/python -m src.runner -s 1 --strategy fast_enumerative --max-depth 4 --max-complexity 4  - <<EOF
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
## Command Line Options

The following command line options are available:  
`-s, --sygus-standard:` Specify the SyGuS language standard used in the input file. Accepted values are:
- `1`: SyGuS language standard version 1.
- `2`: SyGuS language standard version 2 (default).

`-v, --verbose`  
  Set the debugging message suppression level of the program output. Accepted values are:
- `0`: No suppression; all output is printed to the console (default).
- `1`: Suppress some warnings.
- `2`: Suppress all output except success/failure messages.

`input_file`: Path to the input file containing the synthesis problem.   
- If `-` is provided, the program will read the input from STDIN.

`--strategy`: Choose the synthesis strategy. Accepted values are:  
- `fast_enumerative`: Fast enumerative strategy (default).  
- `random_search`: Random search strategy.


`--min-const`: Minimum integer constant value to include in expressions for candidate synthesis.    
`--max-const`: Maximum integer constant value to include in expressions for candidate synthesis.  
`--max-depth`: Maximum integer depth for candidate generation.    
`--max-candidates-at-each-depth`: Maximum number of candidates to evaluate at each depth for the random search strategy.  
`--max-complexity`: Maximum integer complexity for the random search strategy.  
`--random-seed-behaviour`: Behaviour for the random seed for the solver. Accepted values are:  
- `fixed`: Use the provided random seed on every iteration.  
- `random`: Generate a new random seed for each iteration.  
`--random-seed`: Random seed for the random search strategy (used when `--random-seed-behaviour` is set to `fixed`).  
`--randomise-each-iteration`: Randomise the random seed for each iteration in the random search strategy.

## Sample Commands
Here are a few sample commands to run the various strategies:

- Fast Enumerative Strategy with default parameters:
```shell
./venv/bin/python -m src.runner input.sl
```
- Fast Enumerative Strategy with some parameters:
```shell
./venv/bin/python -m src.runner input.sl --strategy fast_enumerative --min-const -5 --max-const 5
```
- Fast Enumerative Strategy with some parameters:
```shell
./venv/bin/python -m src.runner input.sl --strategy random_search --randomise-each-iteration
```



## Run Tests
```shell
make test
```