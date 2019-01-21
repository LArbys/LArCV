#!/bin/bash

#  /!\
# CAVEAT :
# Very specific to Adrien's architecture, no need to try and use it yourself, this WILL not work

setuBooNE

#DataPATH="/Volumes/DataStorage/DeepLearningData/MCC8_Overlays"
DataPATH="~/Desktop/events4Janet"
outputPATH="~/Desktop/events4Janet/displays"
SelectionPATH="~/Desktop/events4Janet"
#for run in {6860..7150}
for run in {7014..8316}
do
    for subrun in {0..2000}
    do
        echo $run":"$subrun
        printf -v larliteFile  "tracker_reco_%04d%04d*.root" "$run" "$subrun"
        printf -v analysisFile "tracker_anaout_%04d%04d*.root" "$run" "$subrun"
        printf -v image2DFile  "supera-Run%06d-SubRun%06d*.root" "$run" "$subrun"
        if [ -f  $DataPATH/$image2DFile ]; then
            if [ -f $DataPATH/$larliteFile ]; then
#                time python mac/run_display.py $DataPATH/$image2DFile $DataPATH/$larliteFile $DataPATH/$analysisFile $SelectionPATH/png $SelectionPATH/SelectedVerticesFile.txt
#                time python mac/run_display.py $DataPATH/supera/$image2DFile $DataPATH/tracker_reco/$larliteFile $DataPATH/tracker_anaout/$analysisFile $outputPATH $SelectionPATH/selected_1e1p_nueint_v2.txt
                time python mac/run_display.py $DataPATH/supera/$image2DFile $DataPATH/tracker_reco/$larliteFile $DataPATH/tracker_anaout/$analysisFile $outputPATH
            fi
        fi
    done
done
