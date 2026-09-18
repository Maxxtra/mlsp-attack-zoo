#!/bin/bash
# varianta mica a matricei de transfer: n=20
cd "$(dirname "$0")/.."

python3 src/transfer.py --dataset imagenette --eps 4/255 --n 20
