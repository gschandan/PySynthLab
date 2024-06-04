# Dissertation Project

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

## Run Program

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
./venv/bin/python -m src.runner -s 1 - <<EOF
(set-logic LIA)
(synth-fun f ((x Int) (y Int)) Int)
(declare-var x Int)
(declare-var y Int)
(constraint (= (f x y) (f y x)))
(constraint (and (<= x (f x y)) (<= y (f x y))))
(check-synth)
EOF
```
## Command Line Options

The following command line options are available:  
`-s, --sygus-standard:` Specify the SyGuS language standard used in the input file. Accepted values are:
- `1`: SyGuS language standard version 1.
- `2`: SyGuS language standard version 2 (default).

`-v, --verbose`  
  Set the verbosity level of the program output. Accepted values are:
- `0`: No suppression; all output is printed to the console (default).
- `1`: Suppress warnings.
- `2`: Suppress all output except success/failure messages.

`input_file`: Path to the input file containing the synthesis problem.   
If `-` is provided, the program will read the input from STDIN.



## Run Tests
```shell
make test
```