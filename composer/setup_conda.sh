#!/bin/bash

echo "🚀 Setting up Composer Interview Training Environment (using Conda)..."

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "❌ Error: conda is not installed or not in PATH."
    echo "   Please install Anaconda or Miniconda first."
    exit 1
fi

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check if conda environment already exists
if conda env list | grep -q "^composer "; then
    echo "⚠️  Conda environment 'composer' already exists."
    read -p "Do you want to remove and recreate it? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        conda env remove -n composer -y
    else
        echo "Using existing environment..."
        conda activate composer
        exit 0
    fi
fi

echo "📦 Creating conda environment 'composer'..."
conda create -n composer python=3.11 -y

if [ $? -ne 0 ]; then
    echo "❌ Failed to create conda environment."
    exit 1
fi

echo "🔧 Activating conda environment..."
eval "$(conda shell.bash hook)"
conda activate composer

echo "📥 Installing dependencies with conda..."
conda install -y numpy pandas scikit-learn scipy pytest tqdm

echo "📥 Installing additional packages with pip..."
pip install memory-profiler line-profiler pytest-cov

if [ $? -ne 0 ]; then
    echo "⚠️  Some packages may have failed to install, but core dependencies should be ready."
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "📝 To activate the environment in the future, run:"
echo "   conda activate composer"
echo ""
echo "📝 To deactivate when you're done:"
echo "   conda deactivate"
echo ""
echo "🎯 You can now run your exercises! Example:"
echo "   cd 02_data_ml_coding/easy"
echo "   python starter_01.py"
echo ""

