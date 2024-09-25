#!/bin/zsh

# format all python files
find ./visiongraph -name '*.py' -exec autopep8 --in-place '{}' --max-line-length 120 \;