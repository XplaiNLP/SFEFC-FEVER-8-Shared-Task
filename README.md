# Exploring Semantic Filtering Heuristics For Efficient Claim Verification

This repo documents the code for our submission within the [FEVER-8 Shared Task](https://fever.ai/task.html).
The bash, evaluation and submission scripts were included from the [FEVER-8 Baseline repo] (https://github.com/Raldir/FEVER-8-Shared-Task), which is an optimized version of [HerO from the FEVER-7 Shared Task](https://github.com/ssu-humane/HerO).


# Installation
## Preparation
1. Please download the dev knowledge store according to the README.md in the `./knowledge_store` dir.

## Combined Installation and Running with venv
1. `chmod +x ./*.sh`
2. `./install_and_run.sh`


## Separate Installation and Running with venv
1. `chmod +x ./*.sh`
2. `./installation.sh`
3. `source venv/bin/activate`
4. `./run_system.sh`


## Installation and Running with Conda
1. `conda create -n efc python=3.12`
2. `conda activate efc`
3. `export CMAKE_ARGS="-DGGML_CUDA=on" `
4. `python -m pip install llama-cpp-python`
5. `python -m pip install -r requirements.txt`
6. `python -m pip install transformers --upgrade`
7. `python -m pip install sentence_transformers`
8. `chmod +x ./*.sh`
9. `./run_system.sh`