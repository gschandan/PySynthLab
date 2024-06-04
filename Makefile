export PYTHONPATH := $(PWD)
init:
	pip install -r requirements.txt
init_venv:
	./venv/bin/pip install -r requirements.txt 
test:
	pytest