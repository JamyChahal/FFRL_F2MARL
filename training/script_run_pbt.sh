#!/bin/bash

export PYTHONPATH=$PYTHONPATH:~/git/FFRL_F2MARL
source ~/git/FFRL_F2MARL/venv/bin/activate
python3 ~/git/FFRL_F2MARL/training/pbt_tuner_FFRL.py -p ~/git/FFRL_F2MARL/training/params_pbt_FFRL.list -n pbt
