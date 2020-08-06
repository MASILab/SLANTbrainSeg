#!/bin/bash

start=$(date +%s.%N)
# clean up
mkdir /OUTPUTS/outputs

# preprocessing for spleen
/extra/run_Deep_brain_preprocessing
dur=$(echo "$(date +%s.%N) - $start" | bc)
printf "*******preprocessing time: %.6f seconds\n" $dur

start=$(date +%s.%N)
# generate deep segmentation
bash /OUTPUTS/run_all_batches.sh 
dur=$(echo "$(date +%s.%N) - $start" | bc)
printf "*******segmentation time: %.6f seconds\n" $dur

start=$(date +%s.%N)
#postprocessing for spleen
/extra/run_Deep_brain_postprocessing
dur=$(echo "$(date +%s.%N) - $start" | bc)
printf "*******postprocessing time: %.6f seconds\n" $dur

start=$(date +%s.%N)
#generate pdf 
/extra/generate_light_PDF
dur=$(echo "$(date +%s.%N) - $start" | bc)
printf "*******generating pdf time: %.6f seconds\n" $dur

start=$(date +%s.%N)
#generate txt file (ROI vs. volume)
/extra/generate_volume_stat
dur=$(echo "$(date +%s.%N) - $start" | bc)
printf "*******generating text file time: %.6f seconds\n" $dur


#clean up
# rm -r /OUTPUTS/Data_2D
# rm -r /OUTPUTS/DeepSegResults
# rm -r /OUTPUTS/dicom2nifti
# rm -r /OUTPUTS/FinalSeg
# rm -r /OUTPUTS/FinalResult/tmp
