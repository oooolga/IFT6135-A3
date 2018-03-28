#!/bin/sh
export PYTHONUNBUFFERED=1
PYTHON_BIN="/u/luyuchen/miniconda2/envs/pytorch3/bin/python"
$PYTHON_BIN main.py --model lstm_ntm --logdir lstmntm_logs
