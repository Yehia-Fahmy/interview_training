.PHONY: setup lab quickcheck clean

VENV=.venv
PYTHON=$(VENV)/bin/python
PIP=$(VENV)/bin/pip
JUPYTER=$(VENV)/bin/jupyter

setup:
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PYTHON) -m ipykernel install --user --name=interview_training || true

lab:
	$(JUPYTER) lab

quickcheck:
	@echo "Running lightweight checks..."
	@$(PYTHON) -c "\
try:\n\
    import utils.grading as grading\n\
    print('utils.grading available ✅')\n\
except Exception as e:\n\
    print('utils.grading not ready:', e)\n\
"

clean:
	rm -rf $(VENV)

