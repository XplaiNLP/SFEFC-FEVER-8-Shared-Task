#!/bin/bash

source venv/bin/activate

# Install packages
export CMAKE_ARGS="-DGGML_CUDA=on" 
python -m pip install llama-cpp-python
python -m pip install -r requirements.txt
python -m pip install transformers --upgrade
python -m pip install sentence_transformers