#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo "python stage0_dump_txt.py $1 $2"
python ${DIR}/stage0_dump_txt.py $1 $2

echo "python stage1_filter_larcv.py $3 $4"
python ${DIR}/stage1_filter_larcv.py $3 $4

echo "python stage2_hadd.py $5 $6"
python ${DIR}/stage2_hadd.py $5 $6
