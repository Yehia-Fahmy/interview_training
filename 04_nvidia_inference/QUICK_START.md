# Quick Start Guide

Get started with NVIDIA Inference Engineer interview preparation in 5 minutes.

## 👉 Interview Preparation (1-2 weeks)?

**If you're preparing for an upcoming interview, use the focused guide instead:**

- **Read**: `FOCUSED_INTERVIEW_PREP.md` - Critical analysis of what NVIDIA actually tests
- **Practice**: `FOCUSED_EXERCISES.md` - Top 5 must-do exercises
- **Focus**: PyTorch 2.0 ecosystem, quantization, system design (80% of interview)

The guide below is for comprehensive learning over 4-8 weeks.

---

## Choose Your Timeline (Comprehensive Learning)

### Intensive (2 weeks) - For experienced candidates
- **Week 1**: PyTorch ecosystem, optimization techniques, CUDA basics
- **Week 2**: High-performance kernels, TensorRT, distributed inference, profiling

### Standard (4 weeks) - Recommended
- **Week 1**: PyTorch ecosystem & model optimization
- **Week 2**: GPU architecture & CUDA programming
- **Week 3**: High-performance kernels & TensorRT
- **Week 4**: Distributed inference, profiling, system design

### Extended (6-8 weeks) - For thorough preparation
- Deep dives into each area
- Build portfolio projects
- Contribute to open-source
- Multiple mock interviews

## Day 1: Setup & Overview (2 hours)

1. **Read the roadmap** (`roadmap.md`)
2. **Review this guide**
3. **Set up environment**:
   ```bash
   # Install PyTorch 2.0+
   pip install torch torchvision
   
   # Install other dependencies (see requirements.txt)
   pip install transformers accelerate
   ```

4. **Start Week 1**: Read `week_01_pytorch/README.md`

## Weekly Structure

Each week follows this pattern:

1. **Read the weekly README** - Overview and learning objectives
2. **Study key concepts** - Follow day-by-day schedule
3. **Complete practice exercises** - Hands-on coding
4. **Review resources** - Documentation, papers, tutorials
5. **Prepare deep dive** - Technical example for interview

## Practice Exercises

- Located in `practice_exercises/` directory
- Start with `ex_01_*` and progress sequentially
- Each exercise has TODOs to implement
- Compare with reference implementations (if available)

## Interview Preparation

Before your interview:

1. **Review `interview_prep/`**:
   - Common questions (`questions.md`)
   - Coding challenges (`coding_challenges.md`)
   - System design templates (`system_design_templates.md`)

2. **Prepare 2-3 technical deep dives**:
   - Model optimization project
   - GPU kernel optimization
   - System design example

3. **Practice explaining concepts**:
   - Record yourself explaining torch.compile
   - Practice system design with timer (45-60 min)
   - Review common questions

## Key Focus Areas

### Must Master
- PyTorch 2.0 compilation stack
- Model optimization (quantization, pruning)
- GPU architecture basics
- TensorRT optimization
- Distributed inference strategies

### Should Know
- CUDA kernel programming
- Triton and CUTLASS
- Performance profiling
- System design patterns

## Success Checklist

Before interview, ensure you can:
- [ ] Explain PyTorch 2.0 compilation stack
- [ ] Implement basic CUDA kernels
- [ ] Describe quantization/pruning trade-offs
- [ ] Profile and optimize GPU performance
- [ ] Design distributed inference systems
- [ ] Discuss TensorRT optimization pipeline
- [ ] Explain attention mechanisms and KV-caching
- [ ] Design scalable software architectures

## Getting Help

- Review `resources/README.md` for study materials
- Check practice exercise READMEs for hints
- Review weekly guides for detailed explanations

## Next Steps

1. Choose your timeline (Intensive/Standard/Extended)
2. Start with Week 1: `week_01_pytorch/README.md`
3. Complete practice exercises as you go
4. Review interview prep materials before interview

Good luck! 🚀

