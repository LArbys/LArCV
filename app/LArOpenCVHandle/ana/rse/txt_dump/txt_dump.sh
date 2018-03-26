#!/bin/bash

echo "python stage0_dump_txt.py $1 $2"
python stage0_dump_txt.py $1 $2

echo "python stage1_filter_larcv.py $3 $4"
python stage1_filter_larcv.py $3 $4

echo "python stage2_hadd.py $5 $6"
python stage2_hadd.py $5 $6
