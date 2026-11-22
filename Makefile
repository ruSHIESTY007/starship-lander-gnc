.PHONY: all fsw test venv clean
PY ?= .venv/bin/python

all: fsw

fsw:
	$(MAKE) -C fsw lib

test:
	$(MAKE) -C fsw test
	$(PY) -m pytest -q tests

venv:
	python3 -m venv .venv && .venv/bin/pip install -q -r requirements.txt

clean:
	$(MAKE) -C fsw clean
