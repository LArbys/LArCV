#!/bin/bash

#  /!\
# CAVEAT :
# Very specific to Adrien's architecture, no need to try and use it yourself, this WILL not work

#setuBooNE

DataPATH="/Users/hourlier/Documents/PostDocMIT/Research/MicroBooNE/FinalFiles/test21/overlays/OverlayNumuData"
outputPATH=$DataPATH"/displays"

superaPATH=$DataPATH"/supera"
larlitePATH=$DataPATH"/tracker_reco"
anaoutPATH=$DataPATH"/tracker_anaout"
#SelectionPATH="~/Desktop/events4Janet"
for run in {6866..7150}
#for run in {6866..6866}
#for run in {7000..8316}
do
    for subrun in {0..2000}
    do
        echo $run":"$subrun
        printf -v larliteFile  "tracker_reco_%04d%04d*.root" "$run" "$subrun"
        printf -v analysisFile "tracker_anaout_%04d%04d*.root" "$run" "$subrun"
        printf -v image2DFile  "supera-Run%06d-SubRun%06d_*.root" "$run" "$subrun"
#	echo $DataPATH/tracker_reco/$larliteFile
#	echo $DataPATH/tracker_anaout/$analysisFile
#	echo $DataPATH/supera/$image2DFile
        echo $larlitePATH/$larliteFile
        echo $anaoutPATH/$analysisFile
        echo $superaPATH/$image2DFile
#        if [ -e  $superaPATH/$image2DFile ]; then
#	    echo supera OK
#            if [ -f $larlitePATH/$larliteFile ]; then
#                time python mac/run_display.py $DataPATH/$image2DFile $DataPATH/$larliteFile $DataPATH/$analysisFile $SelectionPATH/png $SelectionPATH/SelectedVerticesFile.txt
#                time python mac/run_display.py $DataPATH/supera/$image2DFile $DataPATH/tracker_reco/$larliteFile $DataPATH/tracker_anaout/$analysisFile $outputPATH $SelectionPATH/selected_1e1p_nueint_v2.txt
                time python mac/run_display.py $superaPATH/$image2DFile $larlitePATH/$larliteFile $anaoutPATH/$analysisFile $outputPATH
#            fi
#        fi
    done
done
