#!/bin/bash
# varianta mica a lui carlini-wagner: un model, n=20
cd "$(dirname "$0")/.."

echo "--- efficientnet cw l2 n=20"
python3 src/run_attack.py --dataset imagenette --model efficientnet --attack cw --norm l2 --eps 8/255 --n 20 --steps 100
