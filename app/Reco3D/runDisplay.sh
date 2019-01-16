#!/bin/bash

#  /!\
# CAVEAT :
# Very specific to Adrien's architecture, no need to try and use it yourself, this WILL not work

setuBooNE

DataPATH="/Volumes/DataStorage/DeepLearningData/MCC8_Overlays"
SelectionPATH="/Users/hourlier/Documents/PostDocMIT/Research/MicroBooNE/FinalFiles/test20/overlays"
#for run in {6860..7150}
for run in {7014..7150}
do
    for subrun in {0..2000}
    do
        echo $run":"$subrun
        printf -v larliteFile "tracker_reco_%04d%04d.root" "$run" "$subrun"
        printf -v analysisFile "tracker_anaout_%04d%04d.root" "$run" "$subrun"
        printf -v image2DFile "supera-Run%06d-SubRun%06d.root" "$run" "$subrun"
        if [ -f  $DataPATH/$image2DFile ]; then
            if [ -f $DataPATH/$larliteFile ]; then
#                time python mac/run_display.py $DataPATH/$image2DFile $DataPATH/$larliteFile $DataPATH/$analysisFile $SelectionPATH/png $SelectionPATH/SelectedVerticesFile.txt
                time python mac/run_display.py $DataPATH/$image2DFile $DataPATH/$larliteFile $DataPATH/$analysisFile $SelectionPATH/png4Jarrett $SelectionPATH/ThanksForMakingMePlotsAdrien.txt
            fi
        fi
    done
done
