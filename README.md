# 8090 ML Interview Prep (Jupyter)

This repo provides a guided, notebook-first prep aligned to the 8090 Improvement Engineer interview process.

- Role context: [Job posting](https://docs.google.com/forms/d/e/1FAIpQLSecO-saGzOC6P54ZZ6rqWJT5lS3befP48t9-04aAbm7kVIWHw/viewform), [Company site](https://www.8090.ai)

## Setup

```bash
make setup
make lab
```

## Roadmap

- 01 Code Challenge
  - 01a Arrays/Hashing (warmup)
  - 01b Streaming/Iterators (core)
  - 01c Performance & Memory (advanced)
- 02 Data/ML Coding
  - 02a LLM Offline Evaluation (core)
  - 02b Agent Evaluation Harness (core)
  - 02c A/B Testing Simulation (advanced)
  - 02d Monitoring: Drift & Calibration (advanced)
  - 02e Small ETL + Classical ML (warmup)
- 03 System Design (prompts)
  - 03a LLM System Design
  - 03b Observability & Reliability Tradeoffs

## Timing (Interview Simulation)
- Code Challenge: pick 01b or 01c (45–60m)
- Data/ML Coding: 02a + 02b (75–90m)
- System Design: 03a (45m)

## Rubrics & Checks
- Autograder-style cells verify correctness/perf.
- Rubrics: `rubrics/llm_eval.json`, `rubrics/system_design_checklist.md`.

## Offline-Friendly
- Uses small fixtures in `data/`.

## Makefile
- `make setup` create venv and install deps
- `make lab` launch Jupyter Lab
- `make quickcheck` sanity import checks


