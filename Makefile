export PYTHONPATH := $(PWD)
init:
	pip install -r requirements.txt
test:
	pytest