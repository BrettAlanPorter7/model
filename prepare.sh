#! /bin/bash

python3 -m venv .venv --system-site-packages

source .venv/bin/activate

pip install .

python3 -m src
