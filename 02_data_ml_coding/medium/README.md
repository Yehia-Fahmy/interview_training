# Data/ML Coding - Medium Exercises

These exercises focus on practical ML challenges that require thoughtful problem-solving, feature engineering, and understanding of ML concepts. They are designed to prepare you for interviews where AI assistance is allowed, emphasizing code quality, design rationale, and technical depth.

## Exercises

All exercises are documented in **[EXERCISES.md](EXERCISES.md)**:

1. **Feature Engineering & Selection Pipeline** - Build a comprehensive pipeline for handling missing values, creating derived features, encoding categoricals, and selecting important features
2. **Handling Imbalanced Classification** - Implement sampling strategies, appropriate evaluation metrics, and probability calibration for imbalanced data
3. **Time Series Forecasting Pipeline** - Create a forecasting system with proper feature engineering, multiple models, and time-aware evaluation
4. **Model Interpretability & Explainability** - Build an interpretability system with SHAP values, feature importance, and model auditing
5. **PyTorch Deep Learning Pipeline** - Build a complete PyTorch pipeline with custom datasets, neural network architectures, training loops, and evaluation

## Getting Started

1. **Set up your environment**:
   ```bash
   pip install -r requirements.txt
   ```
   See **[SETUP.md](SETUP.md)** for detailed setup instructions.

2. Read **[EXERCISES.md](EXERCISES.md)** for detailed problem descriptions and requirements

3. Start with `starter_01.py` and work through each exercise

4. Run `python test_all.py` to test your implementations

5. Review the solutions in `solution_*.py` files after attempting each exercise

## Key Focus Areas

- **Code Quality**: Clean, readable, well-documented code
- **Design Rationale**: Understanding and explaining your choices
- **Production Considerations**: Scalability, maintainability, edge cases
- **ML Fundamentals**: Deep understanding of concepts, not just implementation

## Interview Context

These exercises are designed for interviews where:
- **AI assistance is allowed** (e.g., using Cursor/IDE)
- **Code quality matters** more than just getting it working
- **Explaining your design** is as important as the code itself
- **Production readiness** is evaluated

Focus on understanding the concepts, making thoughtful design decisions, and being able to explain your rationale.
