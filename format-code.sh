#!/bin/zsh

# List of directories to search for Python files
folders=(./visiongraph ./examples ./scripts ./snippets ./tests ./tools)

# Loop through each folder and format Python files using autopep8
for folder in "${folders[@]}"
do
  echo "formatting $folder..."
  find "$folder" -name '*.py' -exec autopep8 --in-place '{}' --max-line-length 120 \;
done
