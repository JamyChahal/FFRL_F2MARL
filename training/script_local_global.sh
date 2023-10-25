#!/bin/bash

source ~/.bashrc

export PYTHONPATH=$PYTHONPATH:~/git/FFRL_F2MARL
source ~/git/FFRL_F2MARL/venv/bin/activate
python3 ~/git/FFRL_F2MARL/training/pbt_tuner_F2MARL.py -p ~/git/FFRL_F2MARL/training/params_pbt.list \
-n pbt_f2marl_local_global -r 'glo' -m 'local_global' -w 64 -s 6
