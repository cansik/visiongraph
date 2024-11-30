#!/bin/zsh

# Check if pyment is installed
if ! command -v pyment &> /dev/null; then
  echo "Error: pyment is not installed. Install it with 'pip install pyment'."
  exit 1
fi

# Directory to search for Python files
TARGET_DIR="${1:-.}"

# Find all Python files in the directory and its subdirectories
echo "Searching for Python files in ${TARGET_DIR}..."
PYTHON_FILES=$(find "$TARGET_DIR" -type f -name "*.py")

# Check if any Python files are found
if [ -z "$PYTHON_FILES" ]; then
  echo "No Python files found in ${TARGET_DIR}."
  exit 0
fi

# Process each Python file with pyment
for file in $PYTHON_FILES; do
  echo "Processing $file with pyment..."
  pyment -w "$file"
done

echo "All Python files processed successfully."
