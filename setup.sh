#!/bin/bash

echo "🚀 Setting up Composer Interview Training Environment..."

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python 3 is not installed. Please install Python 3.8+ first."
    exit 1
fi

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check if virtual environment already exists
if [ -d "venv" ]; then
    echo "⚠️  Virtual environment already exists. Activating existing environment..."
else
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
    
    if [ $? -ne 0 ]; then
        echo "❌ Failed to create virtual environment."
        exit 1
    fi
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📥 Installing dependencies..."
pip install -r requirements.txt

if [ $? -ne 0 ]; then
    echo "❌ Failed to install dependencies."
    exit 1
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "📝 To activate the environment in the future, run:"
echo "   source venv/bin/activate"
echo ""
echo "📝 To deactivate when you're done:"
echo "   deactivate"
echo ""
echo "🎯 You can now run your exercises! Example:"
echo "   cd 02_data_ml_coding/easy"
echo "   python starter_01.py"
echo ""

