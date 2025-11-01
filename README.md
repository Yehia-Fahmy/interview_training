## Interview Training Hub

A structured, hands-on workspace to prepare for software engineering and ML interviews. It brings together three complementary tracks:

- **claude**: End-to-end practice sets with worked solutions and study guides.
- **composer**: Concept-first exercises (prompts and scenarios) to deepen fundamentals.
- **gpt**: Runnable Python pipelines, tests, and templates to build and evaluate solutions.

Use this repo to practice coding challenges, ML coding, and system design with increasing difficulty, while also learning how to structure high-quality solutions and evaluations.

---

### Repository structure

- **claude/**: Curated practice with problem + solution pairs
  - `01_code_challenge/`:
    - `easy/`, `medium/`, `hard/` Python problems with matching `_solution.py` files
    - `README.md` with tips per difficulty
  - `02_data_ml_coding/`:
    - `fundamentals/`, `llm_applications/` exercises with solution files
  - `03_system_design/`:
    - Scenarios and case prompts (e.g., `scenarios/01_llm_chatbot.md`)
  - Study aids: `PROGRESS.md`, `QUICKSTART.md`, `START_HERE.md`, `SUMMARY.md`, `resources/`

- **composer/**: Concept exercises and prompts to reason through
  - `01_code_challenge/` (easy/medium/hard): algorithm/data-structure reasoning tasks
  - `02_data_ml_coding/` (easy/medium/hard): ML practice prompts
  - `03_system_design/`: distributed systems and ML systems design exercises
  - Each folder includes a `README.md` and scenario-style markdown tasks

- **gpt/**: Runnable scaffolds and templates
  - `code_challenge/`:
    - `starter.py` per difficulty + `NOTES_TEMPLATE.md` to capture approach
  - `data_ml_coding/`:
    - `foundational/`, `intermediate/`, `advanced/` Python packages
    - Pipelines (`pipeline.py`), evaluation utilities, and example data (`advanced/data/`)
    - Tests (e.g., `intermediate/tests/test_retrieval.py`)
  - `system_design/`: structured notes/templates for design responses

---

### Who is this for?

- Software engineers, data scientists, and ML engineers preparing for interviews
- Practitioners who want both conceptual depth and hands-on practice
- Anyone who prefers having runnable code templates alongside reasoning exercises

---

### Prerequisites

- Python 3.9+ recommended
- `pip` (or `uv`/`pipx`/`poetry` if preferred)
- macOS/Linux or WSL2 on Windows

Optional (recommended):
- Virtual environments (`python -m venv .venv`)

---

### Setup

```bash
# from the repo root
python -m venv .venv
source .venv/bin/activate  # on macOS/Linux
pip install -r claude/requirements.txt  # curated exercises may reference these
```

Notes:
- The `gpt/` projects are self-contained and avoid heavy dependencies by default.
- Some ML exercises may require installing extra packages as noted in their local `README.md` files.

---

### Quick start by track

- **Coding Challenges (fast reps)**
  - Browse: `claude/01_code_challenge/`
  - Try to solve in files without `_solution.py`, then compare against `*_solution.py`.
  - For runnable scaffolds, use `gpt/code_challenge/<difficulty>/starter.py`.

- **Data/ML Coding (pipelines & evaluation)**
  - Start with `gpt/data_ml_coding/foundational/` or `intermediate/` and open `pipeline.py`.
  - Run tests where available (e.g., retrieval tests in `intermediate/tests/`).
  - Cross-reference conceptual prompts in `composer/02_data_ml_coding/` and solved examples in `claude/02_data_ml_coding/`.

- **System Design (reasoning & structure)**
  - Read prompts in `composer/03_system_design/` and `claude/03_system_design/`.
  - Capture answers using templates in `gpt/system_design/RESPONSE_TEMPLATE.md`.

---

### Suggested practice flow

1) Pick a lane for today (coding, ML coding, or system design).
2) Attempt an unsolved exercise (composer → gpt scaffold), timebox yourself.
3) Compare with solutions (claude), note gaps in `NOTES_TEMPLATE.md`.
4) Repeat with a higher difficulty or a different domain.

Tip: Track progress in `claude/PROGRESS.md` and summarize learnings in `claude/SUMMARY.md`.

---

### Running tests (example)

Some tracks include tests. For example, in `gpt/data_ml_coding/intermediate/`:

```bash
cd gpt/data_ml_coding/intermediate
python -m pytest -q
```

---

### Included resources

- `claude/resources/`: company research, interview tips
- `composer/resources/`: conceptual guides (ML concepts, systems patterns)
- `gpt/resources.md`: consolidated references and roadmap

---

### Contributing (personal use or team)

- Keep your own solutions separate from `_solution.py` files.
- Use the provided `NOTES_TEMPLATE.md` to document approach, complexity, trade-offs.
- If collaborating, standardize on a Python version and testing approach.

---

### License

This repository is intended for personal interview preparation. Add a license if you plan to share or publish your solutions.

---

### Acknowledgments

Structure inspired by combining practice prompts, runnable scaffolds, and worked solutions for efficient, outcome-driven interview preparation.


