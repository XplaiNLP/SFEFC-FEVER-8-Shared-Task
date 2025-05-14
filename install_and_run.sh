#!/bin/bash
set -e

python3 -m venv venv

source venv/bin/activate
export VIRTUAL_ENV="venv"

source ./installation.sh
source ./run_system.sh