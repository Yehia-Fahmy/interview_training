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
	@$(PYTHON) - <<'EOF'
try:
    import utils.grading as grading  # noqa: F401
    print("utils.grading available ✅")
except Exception as e:
    print("utils.grading not ready:", e)
EOF

clean:
	rm -rf $(VENV)

