#! /bin/bash

python -m venv .venv --system-site-packages

source .venv/bin/activate

pip install .[pi]

python -m src
