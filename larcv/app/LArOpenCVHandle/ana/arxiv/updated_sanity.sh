#!/bin/bash

echo "START"
echo ""
echo "python track_shower.py $1 $2 1>$1_eff.txt"
echo ""
python track_shower.py $1 $2 1>$1_eff.txt
echo "Done"
echo ""
