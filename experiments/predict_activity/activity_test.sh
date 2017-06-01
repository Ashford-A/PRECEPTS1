#!/bin/bash

source activate HetMan
cd ~/compbio/scripts
python HetMan/experiments/baseline/baseline_test.py "$@"

