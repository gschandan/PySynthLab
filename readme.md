# Dissertation Project

This project is a synthesis engine for solving LIA synthesis problems using CEGIS. 

## Prerequisites

- Python 3.8 or higher
- Virtualenv

## Installation

1. Clone the repository:

```shell
git clone git@github.com:gschandan/PySynthLab.git
cd Dissertation
```

2. Create a virtual environment:

```shell
python -m venv venv
```

3. Activate the virtual environment:

```shell
source venv/bin/activate
```
4. Install the requirements:
```shell
pip install -r requirements.txt
```
or alternatively using Make:

```shell
make init
```

## Options



## Run Program

Either provide a path to the sygus input file:
```shell
python src/runner.py -s 1 /path_to_file/small.sl  
```
or provide the input problem via STDIN:
```shell
python -m src.runner -s 1 - <<EOF                                                                                                                                                                                                                                                         2 ↵ gii@gii
(set-logic LIA)
(synth-fun f ((x Int) (y Int)) Int)
(declare-var x Int)
(declare-var y Int)
(constraint (= (f x y) (f y x)))
(constraint (and (<= x (f x y)) (<= y (f x y))))
(check-synth)
EOF

```

## Run Tests
```shell
make test
```