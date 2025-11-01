# 8090 AI Interview Preparation Repository

Welcome to your personalized interview preparation guide for the **AI Improvement Engineer** position at 8090 Solutions Inc.

## 📋 About 8090 AI

8090 Solutions Inc. is building a revolutionary "Software Factory" - an AI-first platform that delivers fully-managed and hosted software, transforming the traditional software development lifecycle. Co-founded by Chamath Palihapitiya, the company focuses on autonomous software development using advanced LLMs and agentic systems.

### The Role: AI Improvement Engineer

You'll be working on:
- Deploying and managing LLMs and advanced ML systems in production
- Designing and executing model evaluation strategies
- Building tools for offline experimentation and model validation
- Developing production-grade agentic systems
- Automating ML lifecycle processes
- Testing and QA of agent behaviors

## 🎯 Interview Structure

### 1. **Code Challenge (Python)** 
- **Focus**: Lower-level computing concepts and optimizations
- **AI Usage**: Limited - only for syntax guidance and minor completions
- **Cannot**: Directly ask how to solve the problem
- **Duration**: ~60-90 minutes

### 2. **Data/ML Coding Interview**
- **Focus**: Practical ML implementation and code quality
- **AI Usage**: Fully AI-assisted (Cursor IDE)
- **Evaluation**: Code quality, technical clarity, design rationale
- **Duration**: ~60-90 minutes

### 3. **System Design Interview**
- **Focus**: Distributed applications and ML systems architecture
- **Format**: Discussion-based (not hands-on)
- **Topics**: Scalability, data pipelines, model deployment, monitoring
- **Duration**: ~45-60 minutes

## 📚 Repository Structure

```
├── 01_code_challenge/          # Python optimization problems
│   ├── easy/                   # Warm-up problems
│   ├── medium/                 # Core interview-level problems
│   ├── hard/                   # Advanced optimization challenges
│   └── README.md              # Guide and tips
│
├── 02_data_ml_coding/         # ML implementation problems
│   ├── fundamentals/          # Basic ML algorithms
│   ├── llm_applications/      # LLM-specific tasks
│   ├── evaluation/            # Model evaluation & metrics
│   ├── mlops/                 # Production ML workflows
│   └── README.md             # Guide and tips
│
├── 03_system_design/          # System design scenarios
│   ├── scenarios/             # Design problems
│   ├── templates/             # Design templates
│   └── README.md             # Guide and tips
│
└── resources/                 # Additional materials
    ├── company_research.md    # 8090 AI deep dive
    ├── ml_concepts.md         # Quick reference
    └── interview_tips.md      # Logistics and strategy
```

## 🗓️ 4-Week Study Plan

### Week 1: Python Fundamentals & Optimization
**Goal**: Master Python performance optimization and lower-level concepts

- **Days 1-2**: Memory management, data structures, complexity analysis
- **Days 3-4**: Algorithm optimization, bit manipulation, system-level concepts
- **Days 5-7**: Practice problems (easy → medium)

**Daily Time**: 3-4 hours
- 1 hour: Concept review
- 2-3 hours: Coding practice

### Week 2: ML Fundamentals & Implementation
**Goal**: Implement ML algorithms from scratch and understand internals

- **Days 1-2**: Supervised learning (regression, classification)
- **Days 3-4**: Unsupervised learning (clustering, dimensionality reduction)
- **Days 5-6**: Model evaluation, cross-validation, metrics
- **Day 7**: End-to-end ML pipeline project

**Daily Time**: 4-5 hours
- 1-2 hours: Theory and concepts
- 3 hours: Implementation practice

### Week 3: LLMs & Advanced ML Systems
**Goal**: Work with LLMs and production ML systems

- **Days 1-2**: LLM fundamentals, prompt engineering, fine-tuning
- **Days 3-4**: Agentic systems, tool use, evaluation strategies
- **Days 5-6**: Model monitoring, A/B testing, experimentation
- **Day 7**: Build a mini agentic system

**Daily Time**: 4-5 hours
- 1 hour: Research and reading
- 3-4 hours: Hands-on projects

### Week 4: System Design & Integration
**Goal**: Design scalable ML systems and integrate all knowledge

- **Days 1-2**: ML system design patterns, distributed systems
- **Days 3-4**: Data pipelines, model serving, monitoring
- **Days 5-6**: Practice system design scenarios
- **Day 7**: Mock interviews and review

**Daily Time**: 3-4 hours
- 2 hours: System design practice
- 1-2 hours: Mock interviews and review

## 🎓 Learning Philosophy

### For Code Challenge
- **Think optimization first**: Memory, time complexity, cache efficiency
- **Understand the "why"**: Don't just memorize solutions
- **Practice without AI**: Build muscle memory for core concepts
- **Use AI strategically**: Only for syntax lookup, not problem-solving

### For Data/ML Coding
- **Code quality matters**: Clean, documented, maintainable code
- **Explain your choices**: Articulate why you chose specific approaches
- **Use AI effectively**: Leverage Cursor to write production-quality code
- **Think end-to-end**: Data loading → preprocessing → training → evaluation

### For System Design
- **Start with requirements**: Clarify scale, constraints, use cases
- **Think trade-offs**: No perfect solution, only appropriate ones
- **Go deep on ML specifics**: Model serving, feature stores, monitoring
- **Draw diagrams**: Visual communication is key

## 🚀 Getting Started

1. **Set up your environment**:
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

2. **Start with Week 1**:
```bash
cd 01_code_challenge
cat README.md
```

3. **Track your progress**:
- Use the `progress.md` file in each section
- Mark problems as: Not Started | In Progress | Completed | Reviewed

4. **Review and iterate**:
- Revisit problems you found challenging
- Time yourself on medium/hard problems
- Review solutions and alternative approaches

## 📝 Interview Day Checklist

### Technical Setup
- [ ] Zoom desktop client installed and tested
- [ ] Remote control feature tested
- [ ] Stable internet connection verified
- [ ] Camera and microphone working
- [ ] Backup internet option ready (mobile hotspot)

### Environment Setup
- [ ] Python environment configured
- [ ] Cursor IDE familiar and ready
- [ ] Common libraries installed (numpy, pandas, sklearn, etc.)
- [ ] Code snippets and templates ready

### Mental Preparation
- [ ] Review key concepts the night before
- [ ] Get good sleep
- [ ] Prepare questions about 8090 and the role
- [ ] Have water and snacks ready

## 🎯 Success Metrics

Track your readiness:
- ✅ Can solve 80%+ of medium Python optimization problems in 30-45 min
- ✅ Can implement common ML algorithms from scratch
- ✅ Can design and explain 3+ ML system architectures
- ✅ Comfortable using Cursor/AI tools for rapid development
- ✅ Can articulate design decisions and trade-offs clearly

## 📚 Additional Resources

- [8090 AI Website](https://www.8090.ai)
- [Job Application Form](https://docs.google.com/forms/d/e/1FAIpQLSecO-saGzOC6P54ZZ6rqWJT5lS3befP48t9-04aAbm7kVIWHw/viewform)
- Company Research: `resources/company_research.md`
- ML Quick Reference: `resources/ml_concepts.md`
- Interview Tips: `resources/interview_tips.md`

## 💪 You've Got This!

Remember: 8090 is looking for engineers who can:
1. **Optimize and think deeply** about performance
2. **Build production-quality** ML systems
3. **Design at scale** with thoughtful trade-offs
4. **Communicate clearly** about technical decisions

This repository is your training ground. Work through it systematically, and you'll be well-prepared for your interviews.

Good luck! 🚀

